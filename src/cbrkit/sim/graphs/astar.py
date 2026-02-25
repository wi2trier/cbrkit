import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment

from ...helpers import get_logger, unpack_float
from ...model.graph import Graph, GraphElementType, Node
from ...typing import SimFunc
from .common import GraphSim, SearchGraphSimFunc, SearchState, next_elem, sorted_iter
from .lap import lap_base

__all__ = [
    "HeuristicFunc",
    "SelectionFunc",
    "h1",
    "h2",
    "h3",
    "h4",
    "select1",
    "select2",
    "select3",
    "select4",
    "build",
]

logger = get_logger(__name__)


def node_mapping_feasible[K, N](
    x_node: Node[K, N], y_node: Node[K, N], s: SearchState[K]
) -> bool:
    """Return whether mapping `y_node` to `x_node` is feasible in state `s`.

    A mapping is feasible if it is already fixed in `s.node_mapping` or if both
    nodes are still open candidates (i.e., present in `s.open_x_nodes`/`s.open_y_nodes`).
    """
    return x_node.key == s.node_mapping.get(y_node.key) or (
        y_node.key in s.open_y_nodes and x_node.key in s.open_x_nodes
    )


def feasible_subgraph_pair_sims[K, N, E, G](
    x: Graph[K, N, E, G],
    y: Graph[K, N, E, G],
    s: SearchState[K],
    node_pair_sims: Mapping[tuple[K, K], float],
    edge_pair_sims: Mapping[tuple[K, K], float],
    sub_x: Graph[K, N, E, G],
    sub_y: Graph[K, N, E, G],
) -> tuple[dict[tuple[K, K], float], dict[tuple[K, K], float]]:
    """Filter node/edge pair similarities to encode existing mappings.

    The LAP routines do not know about the current A* state.
    To respect already fixed node mappings while solving on the remaining subgraphs, we:
    - keep only node pairs within `sub_x`/`sub_y`; and
    - keep only edge pairs whose endpoints are consistent with the current mapping,
      i.e., an edge `(y_u,y_v)` may only be paired with `(x_u,x_v)` if for each
      endpoint `y_*` either it is open and `x_*` is open, or it is already mapped
      exactly to `x_*`.
    """

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

        y_edge = y.edges[y_e]
        x_edge = x.edges[x_e]

        if node_mapping_feasible(
            x_edge.source, y_edge.source, s
        ) and node_mapping_feasible(x_edge.target, y_edge.target, s):
            sub_edge_pairs[(y_e, x_e)] = sim

    return sub_node_pairs, sub_edge_pairs


@dataclass(slots=True, frozen=True, order=True)
class PriorityState[K]:
    priority: float
    state: SearchState[K] = field(compare=False)


class HeuristicFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        /,
    ) -> float: ...


class SelectionFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        /,
    ) -> None | tuple[K, GraphElementType]: ...


@dataclass(slots=True, frozen=True)
class h1[K, N, E, G](HeuristicFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        """Heuristic to compute future similarity"""

        return (len(s.open_y_nodes) + len(s.open_y_edges)) / (
            len(y.nodes) + len(y.edges)
        )


@dataclass(slots=True)
class h2[K, N, E, G](HeuristicFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        h_val = 0

        for y_key in s.open_y_nodes:
            h_val += max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in x.nodes.keys()),
                default=0.0,
            )

        for y_key in s.open_y_edges:
            h_val += max(
                (edge_pair_sims.get((y_key, x_key), 0.0) for x_key in x.edges.keys()),
                default=0.0,
            )

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class h3[K, N, E, G](HeuristicFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        h_val = 0

        for y_key in s.open_y_nodes:
            h_val += max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in s.open_x_nodes),
                default=0.0,
            )

        for y_key in s.open_y_edges:
            h_val += max(
                (
                    edge_pair_sims.get((y_key, x_key), 0.0)
                    for x_key in s.open_x_edges
                    if node_mapping_feasible(
                        x.edges[x_key].source, y.edges[y_key].source, s
                    )
                    and node_mapping_feasible(
                        x.edges[x_key].target, y.edges[y_key].target, s
                    )
                ),
                default=0.0,
            )

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class h4[K, N, E, G](HeuristicFunc[K, N, E, G], lap_base[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        # Build subgraphs in a deterministic order to ensure stable LAP results.
        sub_x = Graph(
            nodes=frozendict((k, x.nodes[k]) for k in sorted_iter(s.open_x_nodes)),
            edges=frozendict((k, x.edges[k]) for k in sorted_iter(s.open_x_edges)),
            value=x.value,
        )
        sub_y = Graph(
            nodes=frozendict((k, y.nodes[k]) for k in sorted_iter(s.open_y_nodes)),
            edges=frozendict((k, y.edges[k]) for k in sorted_iter(s.open_y_edges)),
            value=y.value,
        )

        # Early termination for trivial cases
        if len(s.open_y_nodes) == 0 and len(s.open_y_edges) == 0:
            return 0.0

        # Encode current mappings by filtering pair similarities accordingly.
        sub_node_pair_sims, sub_edge_pair_sims = feasible_subgraph_pair_sims(
            x, y, s, node_pair_sims, edge_pair_sims, sub_x, sub_y
        )

        cost_matrix = self.build_cost_matrix(
            sub_x,
            sub_y,
            sub_node_pair_sims,
            sub_edge_pair_sims,
        )
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        cost: float = cost_matrix[row_idx, col_idx].sum()

        # Convert normalized cost to similarity and scale by subgraph size relative to full graph
        full_upper_bound = self.cost_upper_bound(x, y)
        sub_upper_bound = self.cost_upper_bound(sub_x, sub_y)

        if not (full_upper_bound > 0 and sub_upper_bound > 0):
            return 0.0

        similarity = max(0.0, 1.0 - cost)

        # Scale the heuristic by the proportion of remaining elements
        scaling_factor = sub_upper_bound / full_upper_bound

        return similarity * scaling_factor


@dataclass(slots=True, frozen=True)
class select1[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        if s.open_y_nodes:
            return next_elem(s.open_y_nodes), "node"

        if s.open_y_edges:
            return next_elem(s.open_y_edges), "edge"

        return None


@dataclass(slots=True, frozen=True)
class select2[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        edge_candidates = {
            key
            for key in sorted_iter(s.open_y_edges)
            if y.edges[key].source.key not in s.open_y_nodes
            and y.edges[key].target.key not in s.open_y_nodes
        }

        if edge_candidates:
            return next_elem(edge_candidates), "edge"

        if s.open_y_nodes:
            return next_elem(s.open_y_nodes), "node"

        return None


@dataclass(slots=True, frozen=True)
class select3[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

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
                    x.edges[x_key].source, y.edges[y_key].source, s
                )
                and node_mapping_feasible(
                    x.edges[x_key].target, y.edges[y_key].target, s
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
            elif s.open_y_edges:
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

        # select the one with the lowest key
        selection_key, selection_type = next_elem(
            best_selections,
            key=lambda item: item[0],
        )

        if selection_type == "edge":
            edge = y.edges[selection_key]

            if edge.source.key in s.open_y_nodes:
                return edge.source.key, "node"
            elif edge.target.key in s.open_y_nodes:
                return edge.target.key, "node"

        return selection_key, selection_type


@dataclass(slots=True)
class select4[K, N, E, G](SelectionFunc[K, N, E, G], lap_base[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        edge_candidates = {
            key
            for key in sorted_iter(s.open_y_edges)
            if y.edges[key].source.key not in s.open_y_nodes
            and y.edges[key].target.key not in s.open_y_nodes
        }

        if edge_candidates:
            return next_elem(edge_candidates), "edge"

        if s.open_y_nodes:
            if len(s.open_x_nodes) == 0:
                return next_elem(s.open_y_nodes), "node"

            # Build subgraphs in a deterministic order matching the cost-matrix layout.
            sub_x = Graph(
                nodes=frozendict((k, x.nodes[k]) for k in sorted_iter(s.open_x_nodes)),
                edges=frozendict((k, x.edges[k]) for k in sorted_iter(s.open_x_edges)),
                value=x.value,
            )
            sub_y = Graph(
                nodes=frozendict((k, y.nodes[k]) for k in sorted_iter(s.open_y_nodes)),
                edges=frozendict((k, y.edges[k]) for k in sorted_iter(s.open_y_edges)),
                value=y.value,
            )

            # Encode current mappings by filtering pair similarities accordingly.
            sub_node_pair_sims, sub_edge_pair_sims = feasible_subgraph_pair_sims(
                x, y, s, node_pair_sims, edge_pair_sims, sub_x, sub_y
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
            row2y = {r: k for r, k in enumerate(sub_y.nodes.keys())}

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


@dataclass(slots=True)
class build[K, N, E, G](
    SearchGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Performs an A* search as described by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005)

    Args:
        node_sim_func: A function to compute the similarity between two nodes.
        edge_sim_func: A function to compute the similarity between two edges.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        heuristic_func: A heuristic function to compute the future similarity.
        selection_func: A function to select the next node or edge to be mapped.
        init_func: A function to initialize the state.
        beam_width: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.
            Disabled by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).
        pathlength_weight: Favor long partial edit paths over shorter ones.
            Disabled by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).

    Returns:
        The similarity between a query and a case graph along with the mapping.
    """

    heuristic_func: HeuristicFunc[K, N, E, G] = field(default_factory=h3)
    selection_func: SelectionFunc[K, N, E, G] = field(default_factory=select3)
    beam_width: int = 0
    pathlength_weight: int = 0

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> list[SearchState[K]]:
        """Expand a given node and its queue"""

        next_states: list[SearchState[K]] = []
        selection = self.selection_func(
            x,
            y,
            state,
            node_pair_sims,
            edge_pair_sims,
        )

        if selection is None:
            return next_states

        y_key, y_type = selection

        if y_type == "node":
            next_states = self.expand_node(x, y, state, y_key)

        elif y_type == "edge":
            next_states = self.expand_edge(x, y, state, y_key)

        return next_states

    def compute_priority(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        past_sim = unpack_float(
            self.similarity(
                x,
                y,
                state.node_mapping,
                state.edge_mapping,
                node_pair_sims,
                edge_pair_sims,
            )
        )
        future_sim = self.heuristic_func(x, y, state, node_pair_sims, edge_pair_sims)
        prio = 1 - (past_sim + future_sim)

        if self.pathlength_weight > 0:
            # Calculate the number of mapping decisions made so far (partial edit path length)
            # This includes actual mappings plus null mappings (elements processed but not mapped)
            total_y_elements = len(y.nodes) + len(y.edges)
            open_y_elements = len(state.open_y_nodes) + len(state.open_y_edges)
            num_paths = total_y_elements - open_y_elements
            return prio / (self.pathlength_weight**num_paths)

        return prio

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        open_set: list[PriorityState[K]] = []
        best_state = self.init_search_state(x, y)
        # best_similarity = self.similarity(
        #     x,
        #     y,
        #     best_state.node_mapping,
        #     best_state.edge_mapping,
        #     node_pair_sims,
        #     edge_pair_sims,
        # )
        heapq.heappush(open_set, PriorityState(0, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state
            # current_similarity = self.similarity(
            #     x,
            #     y,
            #     current_state.node_mapping,
            #     current_state.edge_mapping,
            #     node_pair_sims,
            #     edge_pair_sims,
            # )

            # not needed because we add null mappings and
            # the first item of the queue is always the best one
            # if current_similarity.value > best_similarity.value:
            #     best_state = current_state
            #     best_similarity = current_similarity

            if self.finished(current_state):
                best_state = current_state
                break

            next_states = self.expand(
                x,
                y,
                current_state,
                node_pair_sims,
                edge_pair_sims,
            )

            for next_state in next_states:
                next_prio = self.compute_priority(
                    x, y, next_state, node_pair_sims, edge_pair_sims
                )
                heapq.heappush(open_set, PriorityState(next_prio, next_state))

            if self.beam_width > 0 and len(open_set) > self.beam_width:
                open_set = heapq.nsmallest(self.beam_width, open_set)
                heapq.heapify(open_set)

        return self.similarity(
            x,
            y,
            best_state.node_mapping,
            best_state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
