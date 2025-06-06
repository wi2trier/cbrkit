import heapq
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from ...helpers import (
    get_logger,
    unpack_float,
)
from ...model.graph import (
    Graph,
    GraphElementType,
    Node,
)
from ...typing import SimFunc
from .common import (
    GraphSim,
    SearchGraphSimFunc,
    SearchState,
)

__all__ = [
    "HeuristicFunc",
    "SelectionFunc",
    "h1",
    "h2",
    "h3",
    "select1",
    "select2",
    "select3",
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

        def mapping_possible(x_node: Node[K, N], y_node: Node[K, N]) -> bool:
            return x_node.key == s.node_mapping.get(y_node.key) or (
                y_node.key in s.open_y_nodes and x_node.key in s.open_x_nodes
            )

        for y_key in s.open_y_edges:
            h_val += max(
                (
                    edge_pair_sims.get((y_key, x_key), 0.0)
                    for x_key in s.open_x_edges
                    if mapping_possible(x.edges[x_key].source, y.edges[y_key].source)
                    and mapping_possible(x.edges[x_key].target, y.edges[y_key].target)
                ),
                default=0.0,
            )

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

        try:
            return next(iter(s.open_y_nodes)), "node"
        except StopIteration:
            pass

        try:
            return next(iter(s.open_y_edges)), "edge"
        except StopIteration:
            pass

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

        try:
            return next(
                key
                for key in s.open_y_edges
                if y.edges[key].source.key not in s.open_y_nodes
                and y.edges[key].target.key not in s.open_y_nodes
            ), "edge"
        except StopIteration:
            pass

        try:
            return next(iter(s.open_y_nodes)), "node"
        except StopIteration:
            pass

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
        pathlength_weight: Add a penalty for states with few mapped elements that already have a low similarity.
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
            node_null_mapping = (
                set(y.nodes.keys())
                - set(state.node_mapping.keys())
                - set(state.open_y_nodes)
            )
            edge_null_mapping = (
                set(y.edges.keys())
                - set(state.edge_mapping.keys())
                - set(state.open_y_edges)
            )
            num_paths = (
                len(state.node_mapping)
                + len(state.edge_mapping)
                + len(node_null_mapping)
                + len(edge_null_mapping)
            )
            return prio / (self.pathlength_weight**num_paths)

        return prio

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        # if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
        #     self_inv = dataclasses.replace(self, _invert=True)
        #     return self.invert_similarity(x, y, self_inv(x=y, y=x))

        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        open_set: list[PriorityState[K]] = []
        best_state = self.init_search_state(x, y)
        heapq.heappush(open_set, PriorityState(0, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state

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
                open_set = open_set[: self.beam_width]

        return self.similarity(
            x,
            y,
            best_state.node_mapping,
            best_state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
