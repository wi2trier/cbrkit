import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from frozendict import frozendict

from ...helpers import (
    get_logger,
    unpack_float,
)
from ...model.graph import (
    Graph,
    GraphElementType,
)
from ...typing import SimFunc
from .common import ElementMatcher, GraphSim, SearchGraphSimFunc, SearchState

__all__ = [
    "HeuristicFunc",
    "SelectionFunc",
    "InitFunc",
    "h1",
    "h2",
    "h3",
    "select1",
    "select2",
    "select3",
    "init1",
    "init2",
    "build",
]

logger = get_logger(__name__)


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
        heuristic_func: HeuristicFunc[K, N, E, G],
        /,
    ) -> None | tuple[K, GraphElementType]: ...


class InitFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
        /,
    ) -> SearchState[K]: ...


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
            max_sim = max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in x.nodes.keys()),
                default=0.0,
            )

            h_val += max_sim

        for y_key in s.open_y_edges:
            max_sim = max(
                (edge_pair_sims.get((y_key, x_key), 0.0) for x_key in x.edges.keys()),
                default=0.0,
            )

            h_val += max_sim

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
            max_sim = max(
                (
                    node_pair_sims.get((y_key, x_key), 0.0)
                    for x_key in x.nodes.keys()
                    if x_key not in s.node_mapping.values()
                ),
                default=0.0,
            )

            h_val += max_sim

        for y_key in s.open_y_edges:
            y_edge = y.edges[y_key]
            max_sim = max(
                (
                    edge_pair_sims.get((y_key, x_key), 0.0)
                    for x_key, x_edge in x.edges.items()
                    if x_key not in s.edge_mapping.values()
                    and y_edge.source.key not in s.node_mapping
                    and y_edge.target.key not in s.node_mapping
                    and x_edge.source.key not in s.node_mapping.values()
                    and x_edge.target.key not in s.node_mapping.values()
                ),
                default=0.0,
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True, frozen=True)
class select1[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: SearchState[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        heuristic_func: HeuristicFunc[K, N, E, G],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        if s.open_y_nodes:
            return next(iter(s.open_y_nodes)), "node"

        if s.open_y_edges:
            return next(iter(s.open_y_edges)), "edge"

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
        heuristic_func: HeuristicFunc[K, N, E, G],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        edge_candidates = (
            key
            for key in s.open_y_edges
            if y.edges[key].source.key not in s.node_mapping
            and y.edges[key].target.key not in s.node_mapping
        )

        try:
            return next(edge_candidates), "edge"
        except StopIteration:
            pass

        if s.open_y_nodes:
            return next(iter(s.open_y_nodes)), "node"

        if s.open_y_edges:
            return next(iter(s.open_y_edges)), "edge"

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
        heuristic_func: HeuristicFunc[K, N, E, G],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        heuristic_scores: list[tuple[K, GraphElementType, float]] = []

        for y_key in s.open_y_nodes:
            heuristic_scores.append(
                (
                    y_key,
                    "node",
                    heuristic_func(x, y, s, node_pair_sims, edge_pair_sims),
                )
            )

        for y_key in s.open_y_edges:
            heuristic_scores.append(
                (
                    y_key,
                    "edge",
                    heuristic_func(x, y, s, node_pair_sims, edge_pair_sims),
                )
            )

        if not heuristic_scores:
            return None

        best_selection = max(heuristic_scores, key=lambda x: x[2])

        selection_key, selection_type, _ = best_selection

        if selection_type == "edge":
            edge = y.edges[selection_key]

            if edge.source.key in s.open_y_nodes:
                return edge.source.key, "node"
            elif edge.target.key in s.open_y_nodes:
                return edge.target.key, "node"

        return selection_key, selection_type


@dataclass(slots=True, frozen=True)
class init1[K, N, E, G](InitFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> SearchState[K]:
        return SearchState(
            frozendict(),
            frozendict(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
            frozenset(x.nodes.keys()),
            frozenset(x.edges.keys()),
        )


@dataclass(slots=True, init=False)
class init2[K, N, E, G](InitFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> SearchState[K]:
        # pre-populate the mapping with nodes/edges that only have one possible legal mapping
        possible_node_mappings: defaultdict[K, set[K]] = defaultdict(set)
        possible_edge_mappings: defaultdict[K, set[K]] = defaultdict(set)

        for y_key, x_key in itertools.product(y.nodes.keys(), x.nodes.keys()):
            if node_matcher(x.nodes[x_key].value, y.nodes[y_key].value):
                possible_node_mappings[y_key].add(x_key)

        for y_key, x_key in itertools.product(y.edges.keys(), x.edges.keys()):
            if edge_matcher(x.edges[x_key].value, y.edges[y_key].value):
                possible_edge_mappings[y_key].add(x_key)

        node_mappings: dict[K, K] = {
            y_key: next(iter(x_keys))
            for y_key, x_keys in possible_node_mappings.items()
            if len(x_keys) == 1
        }
        edge_mappings: dict[K, K] = {
            y_key: next(iter(x_keys))
            for y_key, x_keys in possible_edge_mappings.items()
            if len(x_keys) == 1
        }

        return SearchState(
            frozendict(node_mappings),
            frozendict(edge_mappings),
            frozenset(y.nodes.keys() - node_mappings.keys()),
            frozenset(y.edges.keys() - edge_mappings.keys()),
            frozenset(x.nodes.keys() - node_mappings.values()),
            frozenset(x.edges.keys() - edge_mappings.values()),
        )


@dataclass(slots=True)
class build[K, N, E, G](
    SearchGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        heuristic_func: A heuristic function to compute the future similarity.
        selection_func: A function to select the next node or edge to be mapped.
        init_func: A function to initialize the state.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        beam_width: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.
            Disabled by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).
        pathlength_weight: Add a penalty for states with few mapped elements that already have a low similarity.
            Disabled by default. Based on [Neuhaus et al. (2006)](https://doi.org/10.1007/11815921_17).

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    heuristic_func: HeuristicFunc[K, N, E, G] = field(default_factory=h3)
    selection_func: SelectionFunc[K, N, E, G] = field(default_factory=select3)
    init_func: InitFunc[K, N, E, G] = field(default_factory=init1)
    beam_width: int = 0
    pathlength_weight: int = 0
    allow_case_oriented: bool = True

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
            self.heuristic_func,
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
            num_paths = len(state.node_mapping) + len(state.edge_mapping)
            return prio / (self.pathlength_weight**num_paths)

        return prio

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform an A* analysis of the x base and the y"""
        if (
            len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges)
            and self.allow_case_oriented
        ):
            return self.invert_similarity(x, y, self(x=y, y=x))

        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        open_set: list[PriorityState[K]] = []
        best_state = self.init_func(x, y, self.node_matcher, self.edge_matcher)
        heapq.heappush(open_set, PriorityState(0, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state

            if not current_state.open_y_nodes and not current_state.open_y_edges:
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
                open_set = open_set[: self.beam_width]

        return self.similarity(
            x,
            y,
            best_state.node_mapping,
            best_state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
