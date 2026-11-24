import heapq
import logging
import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated, Literal, Protocol, override

import numpy as np
import scipy
from pydantic import Field

from ..._util import sorted_keys
from ...helpers import frozendict
from ...model.graph import Graph, GraphElementType, Node
from ...model.similarity import GraphSim
from ...schema import Family, serializable
from ._base import BaseGraphEditFunc, PairSimilarities
from ._search import SearchGraphSimFunc, SearchState, next_elem
from .lap import BaseLAP

__all__ = [
    "AssignmentCost",
    "Astar",
    "BestOpenPair",
    "BestPair",
    "EdgesFirst",
    "HeuristicFunc",
    "InOrder",
    "MostConstrained",
    "OpenAssignment",
    "SelectionFunc",
    "Unmapped",
]

logger = logging.getLogger(__name__)


def node_mapping_feasible[K: Hashable, N](
    x_node: Node[K, N], y_node: Node[K, N], s: SearchState[K]
) -> bool:
    """Return whether mapping `y_node` to `x_node` is feasible in state `s`.

    A mapping is feasible if it is already fixed in `s.node_mapping` or if both
    nodes are still open candidates (i.e., present in `s.open_x_nodes`/`s.open_y_nodes`).
    """
    return x_node.key == s.node_mapping.get(y_node.key) or (
        y_node.key in s.open_y_nodes and x_node.key in s.open_x_nodes
    )


def open_subgraphs[K: Hashable, N, E, G](
    case: Graph[K, N, E, G],
    query: Graph[K, N, E, G],
    s: SearchState[K],
    node_pair_sims: Mapping[tuple[K, K], float],
    edge_pair_sims: Mapping[tuple[K, K], float],
) -> tuple[
    Graph[K, N, E, G],
    Graph[K, N, E, G],
    dict[tuple[K, K], float],
    dict[tuple[K, K], float],
]:
    """Restrict both graphs and their pair similarities to the still open elements.

    The subgraphs are built in a deterministic order, since the callers map the rows
    and columns of the resulting cost matrix back to keys by position.

    The LAP routines do not know about the current A* state.
    To respect already fixed node mappings while solving on the remaining subgraphs, we:
    - keep only node pairs within `sub_x`/`sub_y`; and
    - keep only edge pairs whose endpoints are consistent with the current mapping,
      i.e., an edge `(y_u,y_v)` may only be paired with `(x_u,x_v)` if for each
      endpoint `y_*` either it is open and `x_*` is open, or it is already mapped
      exactly to `x_*`.
    """
    sub_x = Graph(
        nodes=frozendict((k, case.nodes[k]) for k in sorted_keys(s.open_x_nodes)),
        edges=frozendict((k, case.edges[k]) for k in sorted_keys(s.open_x_edges)),
        value=case.value,
    )
    sub_y = Graph(
        nodes=frozendict((k, query.nodes[k]) for k in sorted_keys(s.open_y_nodes)),
        edges=frozendict((k, query.edges[k]) for k in sorted_keys(s.open_y_edges)),
        value=query.value,
    )

    # Restrict to nodes contained in the subgraphs (open elements)
    sub_node_pairs: dict[tuple[K, K], float] = {
        (yk, xk): v
        for (yk, xk), v in node_pair_sims.items()
        if yk in sub_y.nodes and xk in sub_x.nodes
    }

    # Restrict to edge pairs that are both in the subgraphs and consistent
    # with the current node mapping for both endpoints.
    sub_edge_pairs: dict[tuple[K, K], float] = {}

    for (y_e, x_e), sim in edge_pair_sims.items():
        if y_e not in sub_y.edges or x_e not in sub_x.edges:
            continue

        y_edge = query.edges[y_e]
        x_edge = case.edges[x_e]

        if node_mapping_feasible(
            x_edge.source, y_edge.source, s
        ) and node_mapping_feasible(x_edge.target, y_edge.target, s):
            sub_edge_pairs[(y_e, x_e)] = sim

    return sub_x, sub_y, sub_node_pairs, sub_edge_pairs


@dataclass(slots=True, frozen=True, order=True)
class PriorityState[K: Hashable]:
    """A search state with a priority value for the A* priority queue.

    Args:
        priority: Priority of the state, smaller is expanded first.
        state: The search state.
    """

    priority: float
    state: SearchState[K] = field(compare=False)


class HeuristicFunc[K: Hashable, N, E, G](Protocol):
    """Estimates the future similarity of unmapped graph elements.

    The returned value is an `unnormalized` upper bound on the similarity that the
    still open query elements can contribute, i.e. it is expressed in the same units
    as the numerator of `BaseGraphSimFunc.similarity`.
    Each pair is weighted by the `node_weight` or `edge_weight` of the given `costs`,
    and the search normalizes the result by its own `cost_upper_bound`, which keeps
    the heuristic admissible for any configuration of the edit costs.
    """

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        costs: BaseGraphEditFunc[K, N, E, G],
        /,
    ) -> float: ...


class SelectionFunc[K: Hashable, N, E, G](Protocol):
    """Selects the next graph element to map during search."""

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        /,
    ) -> None | tuple[K, GraphElementType]: ...


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Unmapped[K: Hashable, N, E, G](HeuristicFunc[K, N, E, G]):
    """Heuristic based on the number of unmapped elements.

    This is the `h_I` heuristic of the paper, where every open query element is
    optimistically assumed to reach the maximum similarity of one.
    """

    kind: Literal["unmapped"] = field(default="unmapped", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        costs: BaseGraphEditFunc[K, N, E, G],
    ) -> float:
        """Heuristic to compute future similarity"""

        return costs.node_weight * len(s.open_y_nodes) + costs.edge_weight * len(
            s.open_y_edges
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class BestPair[K: Hashable, N, E, G](HeuristicFunc[K, N, E, G]):
    """Heuristic based on the best pairwise similarity of unmapped elements."""

    kind: Literal["best_pair"] = field(default="best_pair", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        costs: BaseGraphEditFunc[K, N, E, G],
    ) -> float:
        return costs.node_weight * math.fsum(
            max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in case.nodes),
                default=0.0,
            )
            for y_key in s.open_y_nodes
        ) + costs.edge_weight * math.fsum(
            max(
                (edge_pair_sims.get((y_key, x_key), 0.0) for x_key in case.edges),
                default=0.0,
            )
            for y_key in s.open_y_edges
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class BestOpenPair[K: Hashable, N, E, G](HeuristicFunc[K, N, E, G]):
    """Heuristic based on feasible pairwise similarities of open elements."""

    kind: Literal["best_open_pair"] = field(default="best_open_pair", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        costs: BaseGraphEditFunc[K, N, E, G],
    ) -> float:
        return costs.node_weight * math.fsum(
            max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in s.open_x_nodes),
                default=0.0,
            )
            for y_key in s.open_y_nodes
        ) + costs.edge_weight * math.fsum(
            max(
                (
                    edge_pair_sims.get((y_key, x_key), 0.0)
                    for x_key in s.open_x_edges
                    if node_mapping_feasible(
                        case.edges[x_key].source, query.edges[y_key].source, s
                    )
                    and node_mapping_feasible(
                        case.edges[x_key].target, query.edges[y_key].target, s
                    )
                ),
                default=0.0,
            )
            for y_key in s.open_y_edges
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class OpenAssignment[K: Hashable, N, E, G](
    HeuristicFunc[K, N, E, G], BaseLAP[K, N, E, G]
):
    """Heuristic based on LAP-solved subgraph similarity of open elements."""

    kind: Literal["open_assignment"] = field(default="open_assignment", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        costs: BaseGraphEditFunc[K, N, E, G],
    ) -> float:
        # Nothing is left of the query, so there is no remaining similarity.
        if not s.open_y_nodes and not s.open_y_edges:
            return 0.0

        sub_x, sub_y, sub_node_pair_sims, sub_edge_pair_sims = open_subgraphs(
            case, query, s, node_pair_sims, edge_pair_sims
        )

        # The estimate has to be expressed in the edit costs of the search.
        estimator = replace(
            self,
            node_deletion_cost=costs.node_deletion_cost,
            node_insertion_cost=costs.node_insertion_cost,
            edge_deletion_cost=costs.edge_deletion_cost,
            edge_insertion_cost=costs.edge_insertion_cost,
        )
        cost_matrix = estimator.build_cost_matrix(
            sub_x,
            sub_y,
            sub_node_pair_sims,
            sub_edge_pair_sims,
        )
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
        cost: float = cost_matrix[row_idx, col_idx].sum()

        # `build_cost_matrix` normalizes by the upper bound of the subgraphs,
        # so scaling by it recovers the unnormalized similarity of the remainder.
        sub_upper_bound = estimator.cost_upper_bound(sub_x, sub_y)

        if sub_upper_bound <= 0:
            return 0.0

        return max(0.0, 1.0 - cost) * sub_upper_bound


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class InOrder[K: Hashable, N, E, G](SelectionFunc[K, N, E, G]):
    """Selects the next element in natural order, nodes before edges."""

    kind: Literal["in_order"] = field(default="in_order", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:

        if s.open_y_nodes:
            return next_elem(s.open_y_nodes), "node"

        if s.open_y_edges:
            return next_elem(s.open_y_edges), "edge"

        return None


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class EdgesFirst[K: Hashable, N, E, G](SelectionFunc[K, N, E, G]):
    """Selects edges with both endpoints mapped first, then nodes."""

    kind: Literal["edges_first"] = field(default="edges_first", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:

        edge_candidates = {
            key
            for key in sorted_keys(s.open_y_edges)
            if query.edges[key].source.key not in s.open_y_nodes
            and query.edges[key].target.key not in s.open_y_nodes
        }

        if edge_candidates:
            return next_elem(edge_candidates), "edge"

        if s.open_y_nodes:
            return next_elem(s.open_y_nodes), "node"

        return None


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class MostConstrained[K: Hashable, N, E, G](SelectionFunc[K, N, E, G]):
    """Selects the element with the highest heuristic score and fewest options."""

    kind: Literal["most_constrained"] = field(default="most_constrained", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:

        mapping_options: dict[tuple[K, GraphElementType], int] = {}
        heuristic_scores: dict[tuple[K, GraphElementType], float] = {}

        for y_key in s.open_y_nodes:
            h_vals = [
                node_pair_sims[(y_key, x_key)]
                for x_key in s.open_x_nodes
                if (y_key, x_key) in node_pair_sims
            ]

            if h_vals:
                mapping_options[(y_key, "node")] = len(h_vals)
                heuristic_scores[(y_key, "node")] = max(h_vals)

        for y_key in s.open_y_edges:
            h_vals = [
                edge_pair_sims[(y_key, x_key)]
                for x_key in s.open_x_edges
                if (y_key, x_key) in edge_pair_sims
                and node_mapping_feasible(
                    case.edges[x_key].source, query.edges[y_key].source, s
                )
                and node_mapping_feasible(
                    case.edges[x_key].target, query.edges[y_key].target, s
                )
            ]

            if h_vals:
                mapping_options[(y_key, "edge")] = len(h_vals)
                heuristic_scores[(y_key, "edge")] = max(h_vals)

        if not heuristic_scores:
            # Fallback: select any remaining node or edge for null mapping
            # Use sorted to ensure deterministic selection
            if s.open_y_nodes:
                return next_elem(s.open_y_nodes), "node"

            if s.open_y_edges:
                return next_elem(s.open_y_edges), "edge"
            return None

        # Find the maximum heuristic score
        max_score = max(heuristic_scores.values())
        best_selections = {
            key for key, value in heuristic_scores.items() if value == max_score
        }

        # if multiple selections have the same score, select the one with the lowest number of possible mappings
        if len(best_selections) > 1:
            min_mapping_options = min(mapping_options[key] for key in best_selections)
            best_selections = {
                key
                for key in best_selections
                if mapping_options[key] == min_mapping_options
            }

        # select the one with the lowest key, nodes and edges may share keys
        selection_key, selection_type = next_elem(best_selections)

        if selection_type == "edge":
            edge = query.edges[selection_key]

            if edge.source.key in s.open_y_nodes:
                return edge.source.key, "node"

            if edge.target.key in s.open_y_nodes:
                return edge.target.key, "node"

        return selection_key, selection_type


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AssignmentCost[K: Hashable, N, E, G](
    SelectionFunc[K, N, E, G], BaseLAP[K, N, E, G]
):
    """Selects elements using LAP-based cost analysis of the remaining subgraph."""

    kind: Literal["assignment_cost"] = field(default="assignment_cost", repr=False)

    def __call__(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:

        edge_candidates = {
            key
            for key in sorted_keys(s.open_y_edges)
            if query.edges[key].source.key not in s.open_y_nodes
            and query.edges[key].target.key not in s.open_y_nodes
        }

        if edge_candidates:
            return next_elem(edge_candidates), "edge"

        if s.open_y_nodes:
            if len(s.open_x_nodes) == 0:
                return next_elem(s.open_y_nodes), "node"

            sub_x, sub_y, sub_node_pair_sims, sub_edge_pair_sims = open_subgraphs(
                case, query, s, node_pair_sims, edge_pair_sims
            )

            cost_matrix = self.build_cost_matrix(
                sub_x,
                sub_y,
                sub_node_pair_sims,
                sub_edge_pair_sims,
            )
            # Map row indices back to the actual y keys using exactly the
            # insertion order of the subgraph's nodes to stay aligned with the
            # matrix construction.
            row2y = dict(enumerate(sub_y.nodes))

            # get upper-left quadrant (substitution block only)
            rows = len(sub_y.nodes)
            cols = len(sub_x.nodes)

            if rows == 0 or cols == 0:
                return next_elem(s.open_y_nodes), "node"

            subst_cost_matrix = cost_matrix[:rows, :cols]

            # Select the y-node whose best substitution is cheapest.
            # Break ties by choosing the row with the fewest finite options (most constrained),
            # then deterministically by key.
            row_best_cost = np.min(subst_cost_matrix, axis=1)
            best_cost = np.min(row_best_cost)

            if not np.isfinite(best_cost):
                # No feasible substitutions remain; fall back deterministically.
                return next_elem(s.open_y_nodes), "node"

            candidate_rows = [r for r, c in enumerate(row_best_cost) if c == best_cost]

            if len(candidate_rows) > 1:
                finite_counts = (
                    np.isfinite(subst_cost_matrix[r]).sum() for r in candidate_rows
                )
                min_options = min(finite_counts)
                candidate_rows = [
                    r
                    for r in candidate_rows
                    if np.isfinite(subst_cost_matrix[r]).sum() == min_options
                ]

            # Final deterministic tie-breaker by y-key
            y_candidates = [row2y[r] for r in candidate_rows]
            y_key = next_elem(set(y_candidates))

            return y_key, "node"

        return None


ASTAR_HEURISTICS = Family(
    "astar heuristic",
    (Unmapped, BestPair, BestOpenPair, OpenAssignment),
)


ASTAR_SELECTIONS = Family(
    "astar selection",
    (InOrder, EdgesFirst, MostConstrained, AssignmentCost),
)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Astar[K: Hashable, N, E, G](SearchGraphSimFunc[K, N, E, G]):
    """Performs an A* search as described by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005)

    By default, an element with a legal counterpart must be mapped, which keeps the
    search space smaller.
    Enable `partial_mapping` to also consider leaving it unmapped.
    Together with an admissible `heuristic`, this makes the result optimal unless
    `beam_width` or `pathlength_weight` is used to trade exactness for speed.
    The result is flagged as `exact` only in this case.

    Args:
        heuristic: A heuristic function to compute the future similarity.
        selection: A function to select the next node or edge to be mapped.
        beam_width: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.
            Unbounded by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).
        pathlength_weight: Favor long partial edit paths over shorter ones.
            Disabled by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).
        max_expansions: Number of expanded states after which the search stops and
            returns the most similar mapping among the states it reached, which is
            then not flagged as `exact`.
            Unbounded by default.

    """

    heuristic: Annotated[HeuristicFunc[K, N, E, G], ASTAR_HEURISTICS] = field(
        default_factory=BestOpenPair
    )
    selection: Annotated[SelectionFunc[K, N, E, G], ASTAR_SELECTIONS] = field(
        default_factory=MostConstrained
    )
    beam_width: Annotated[int, Field(ge=1)] | None = None
    pathlength_weight: float = 0.0
    max_expansions: Annotated[int, Field(ge=0)] | None = None
    kind: Literal["astar"] = field(default="astar", repr=False)

    def expand(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> list[SearchState[K]]:
        """Expand a given node and its queue"""

        next_states: list[SearchState[K]] = []
        selection = self.selection(
            case,
            query,
            state,
            node_pair_sims,
            edge_pair_sims,
        )

        if selection is None:
            return next_states

        y_key, y_type = selection

        if y_type == "node":
            next_states = self.expand_node(case, query, state, y_key)

        elif y_type == "edge":
            next_states = self.expand_edge(case, query, state, y_key)

        return next_states

    def compute_priority(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        """Computes the A* priority combining past similarity and heuristic estimate."""
        past_sim = self.similarity_value(
            case,
            query,
            state.node_mapping,
            state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
        upper_bound = self.cost_upper_bound(case, query)
        future_bound = self.heuristic(
            case, query, state, node_pair_sims, edge_pair_sims, self
        )
        future_sim = future_bound / upper_bound if upper_bound > 0 else 0.0
        prio = 1 - (past_sim + future_sim)

        if self.pathlength_weight > 0:
            # Calculate the number of mapping decisions made so far (partial edit path length)
            # This includes actual mappings plus null mappings (elements processed but not mapped)
            total_y_elements = len(query.nodes) + len(query.edges)
            open_y_elements = len(state.open_y_nodes) + len(state.open_y_edges)
            num_paths = total_y_elements - open_y_elements

            return prio / float(self.pathlength_weight) ** num_paths

        return prio

    @override
    def search(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
        state: SearchState[K],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = sims.nodes, sims.edges

        open_set: list[PriorityState[K]] = []
        best_state = state
        best_value = self.similarity_value(
            case,
            query,
            best_state.node_mapping,
            best_state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
        heapq.heappush(open_set, PriorityState(0, best_state))
        goal_reached = False
        expansions = 0

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state

            # Goal testing on pop rather than on push, so the first finished state is
            # the optimal one as long as the heuristic is admissible.
            if self.finished(current_state):
                best_state = current_state
                goal_reached = True
                break

            if self.max_expansions is not None and expansions >= self.max_expansions:
                # Every reached state holds a legal partial mapping.
                for reached in (current_state, *(item.state for item in open_set)):
                    value = self.similarity_value(
                        case,
                        query,
                        reached.node_mapping,
                        reached.edge_mapping,
                        node_pair_sims,
                        edge_pair_sims,
                    )

                    if value > best_value:
                        best_state, best_value = reached, value

                break

            expansions += 1

            next_states = self.expand(
                case,
                query,
                current_state,
                node_pair_sims,
                edge_pair_sims,
            )

            if not next_states:
                # A custom selection function declined to select an element, so this
                # branch dies here and its state is the only result it can offer.
                current_value = self.similarity_value(
                    case,
                    query,
                    current_state.node_mapping,
                    current_state.edge_mapping,
                    node_pair_sims,
                    edge_pair_sims,
                )

                if current_value > best_value:
                    best_state = current_state
                    best_value = current_value

                continue

            for next_state in next_states:
                next_prio = self.compute_priority(
                    case, query, next_state, node_pair_sims, edge_pair_sims
                )
                heapq.heappush(open_set, PriorityState(next_prio, next_state))

            if self.beam_width is not None and len(open_set) > self.beam_width:
                open_set = heapq.nsmallest(self.beam_width, open_set)
                heapq.heapify(open_set)

        return self.similarity(
            case,
            query,
            best_state.node_mapping,
            sims,
            edge_mapping=best_state.edge_mapping,
            exact=goal_reached and self.exact_search,
        )

    @property
    def exact_search(self) -> bool:
        """Whether a completed search is optimal, given an admissible heuristic."""
        return (
            self.partial_mapping
            and self.beam_width is None
            and self.pathlength_weight == 0
        )
