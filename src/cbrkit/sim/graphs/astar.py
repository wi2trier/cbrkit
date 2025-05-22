import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Protocol

from frozendict import frozendict

from ...helpers import (
    batchify_sim,
    get_logger,
    unpack_float,
)
from ...model.graph import (
    Edge,
    Graph,
    GraphElementType,
    Node,
)
from ...typing import AnySimFunc, BatchSimFunc, Float, SimFunc, StructuredValue
from ..wrappers import transpose_value
from .common import ElementMatcher, GraphSim, default_edge_sim, default_element_matcher

__all__ = [
    "PastSimFunc",
    "FutureSimFunc",
    "SelectionFunc",
    "InitFunc",
    "h1",
    "h2",
    "h3",
    "g1",
    "select1",
    "select2",
    "select3",
    "init1",
    "init2",
    "build",
]

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class State[K]:
    # mappings are from y to x
    mapped_nodes: frozendict[K, K]
    mapped_edges: frozendict[K, K]
    remaining_nodes: frozenset[K]
    remaining_edges: frozenset[K]


@dataclass(slots=True, frozen=True, order=True)
class PriorityState[K]:
    priority: float
    state: State[K] = field(compare=False)


@dataclass(slots=True, frozen=True)
class PastSim[K](StructuredValue[float]):
    node_similarities: Mapping[K, float]
    edge_similarities: Mapping[K, float]


class FutureSimFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        /,
    ) -> float: ...


class PastSimFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        /,
    ) -> float | PastSim[K]: ...


class SelectionFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        future_sim_func: FutureSimFunc[K, N, E, G],
        x_centralities: Mapping[K, float],
        y_centralities: Mapping[K, float],
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
    ) -> State[K]: ...


@dataclass(slots=True, frozen=True)
class h1[K, N, E, G](FutureSimFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        """Heuristic to compute future similarity"""

        return (len(s.remaining_nodes) + len(s.remaining_edges)) / (
            len(y.nodes) + len(y.edges)
        )


@dataclass(slots=True)
class h2[K, N, E, G](FutureSimFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        h_val = 0

        for y_key in s.remaining_nodes:
            max_sim = max(
                (node_pair_sims.get((y_key, x_key), 0.0) for x_key in x.nodes.keys()),
                default=0.0,
            )

            h_val += max_sim

        for y_key in s.remaining_edges:
            max_sim = max(
                (edge_pair_sims.get((y_key, x_key), 0.0) for x_key in x.edges.keys()),
                default=0.0,
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class h3[K, N, E, G](FutureSimFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        h_val = 0

        for y_key in s.remaining_nodes:
            max_sim = max(
                (
                    node_pair_sims.get((y_key, x_key), 0.0)
                    for x_key in x.nodes.keys()
                    if x_key not in s.mapped_nodes.values()
                ),
                default=0.0,
            )

            h_val += max_sim

        for y_key in s.remaining_edges:
            y_edge = y.edges[y_key]
            max_sim = max(
                (
                    edge_pair_sims.get((y_key, x_key), 0.0)
                    for x_key, x_edge in x.edges.items()
                    if x_key not in s.mapped_edges.values()
                    and y_edge.source.key not in s.mapped_nodes
                    and y_edge.target.key not in s.mapped_nodes
                    and x_edge.source.key not in s.mapped_nodes.values()
                    and x_edge.target.key not in s.mapped_nodes.values()
                ),
                default=0.0,
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class g1[K, N, E, G](PastSimFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> PastSim[K]:
        """Function to compute the similarity of all previous steps"""

        node_sims = (
            node_pair_sims[(y_key, x_key)] for y_key, x_key in s.mapped_nodes.items()
        )

        edge_sims = (
            edge_pair_sims[(y_key, x_key)] for y_key, x_key in s.mapped_edges.items()
        )

        all_sims = itertools.chain(node_sims, edge_sims)
        total_elements = len(y.nodes) + len(y.edges)

        return PastSim(
            sum(all_sims) / total_elements,
            dict(zip(s.mapped_nodes.keys(), node_sims, strict=True)),
            dict(zip(s.mapped_edges.keys(), edge_sims, strict=True)),
        )


@dataclass(slots=True, frozen=True)
class select1[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        future_sim_func: FutureSimFunc[K, N, E, G],
        x_centralities: Mapping[K, float],
        y_centralities: Mapping[K, float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        if s.remaining_nodes:
            return next(iter(s.remaining_nodes)), "node"

        if s.remaining_edges:
            return next(iter(s.remaining_edges)), "edge"

        return None


@dataclass(slots=True, frozen=True)
class select2[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        future_sim_func: FutureSimFunc[K, N, E, G],
        x_centralities: Mapping[K, float],
        y_centralities: Mapping[K, float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        edge_candidates = (
            key
            for key in s.remaining_edges
            if y.edges[key].source.key not in s.mapped_nodes
            and y.edges[key].target.key not in s.mapped_nodes
        )

        try:
            return next(edge_candidates), "edge"
        except StopIteration:
            pass

        if s.remaining_nodes:
            return next(iter(s.remaining_nodes)), "node"

        if s.remaining_edges:
            return next(iter(s.remaining_edges)), "edge"

        return None


@dataclass(slots=True, frozen=True)
class select3[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        future_sim_func: FutureSimFunc[K, N, E, G],
        x_centralities: Mapping[K, float],
        y_centralities: Mapping[K, float],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        heuristic_scores: list[tuple[K, GraphElementType, float]] = []

        for y_key in s.remaining_nodes:
            heuristic_scores.append(
                (
                    y_key,
                    "node",
                    future_sim_func(x, y, s, node_pair_sims, edge_pair_sims),
                )
            )

        for y_key in s.remaining_edges:
            heuristic_scores.append(
                (
                    y_key,
                    "edge",
                    future_sim_func(x, y, s, node_pair_sims, edge_pair_sims),
                )
            )

        if not heuristic_scores:
            return None

        best_selection = max(heuristic_scores, key=lambda x: x[2])

        selection_key, selection_type, _ = best_selection

        if selection_type == "edge":
            edge = y.edges[selection_key]

            if edge.source.key in s.remaining_nodes:
                return edge.source.key, "node"
            elif edge.target.key in s.remaining_nodes:
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
    ) -> State[K]:
        return State(
            frozendict(),
            frozendict(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
        )


@dataclass(slots=True, init=False)
class init2[K, N, E, G](InitFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> State[K]:
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

        return State(
            frozendict(node_mappings),
            frozendict(edge_mappings),
            frozenset(y.nodes.keys() - node_mappings.keys()),
            frozenset(y.edges.keys() - edge_mappings.keys()),
        )


@dataclass(slots=True)
class build[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        past_sim_func: A heuristic function to compute the similarity of all previous steps.
        future_sim_func: A heuristic function to compute the future similarity.
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

    past_sim_func: PastSimFunc[K, N, E, G]
    future_sim_func: FutureSimFunc[K, N, E, G]
    node_sim_func: InitVar[AnySimFunc[N, Float]]
    node_matcher: ElementMatcher[N]
    edge_sim_func: InitVar[AnySimFunc[Edge[K, N, E], Float] | None] = None
    edge_matcher: ElementMatcher[E] = default_element_matcher
    selection_func: SelectionFunc[K, N, E, G] = field(default_factory=select2)
    init_func: InitFunc[K, N, E, G] = field(default_factory=init1)
    beam_width: int = 0
    pathlength_weight: int = 0
    batch_node_sim_func: BatchSimFunc[Node[K, N], Float] = field(init=False)
    batch_edge_sim_func: BatchSimFunc[Edge[K, N, E], Float] = field(init=False)
    # TODO: Currently not implemented as described in the paper, needs further investigation
    allow_case_oriented_mapping: bool = False

    def __post_init__(
        self,
        any_node_sim_func: AnySimFunc[N, Float],
        any_edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None,
    ) -> None:
        self.batch_node_sim_func = batchify_sim(transpose_value(any_node_sim_func))
        self.batch_edge_sim_func = (
            default_edge_sim(self.batch_node_sim_func)
            if any_edge_sim_func is None
            else batchify_sim(any_edge_sim_func)
        )

    def legal_node_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        return (
            y_key not in state.mapped_nodes.keys()
            and x_key not in state.mapped_nodes.values()
            and self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value)
        )

    def legal_edge_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        x_value = x.edges[x_key]
        y_value = y.edges[y_key]
        mapped_x_source_key = state.mapped_nodes.get(y_value.source.key)
        mapped_x_target_key = state.mapped_nodes.get(y_value.target.key)

        return (
            y_key not in state.mapped_edges.keys()
            and x_key not in state.mapped_edges.values()
            and self.edge_matcher(x_value.value, y_value.value)
            # if the nodes are already mapped, check if they are mapped legally
            and (
                mapped_x_source_key is None or x_value.source.key == mapped_x_source_key
            )
            and (
                mapped_x_target_key is None or x_value.target.key == mapped_x_target_key
            )
        )

    def expand_node(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        y_key: K,
    ) -> list[State[K]]:
        """Expand a given node and its queue"""

        next_states: list[State[K]] = [
            State(
                state.mapped_nodes.set(y_key, x_key),
                state.mapped_edges,
                state.remaining_nodes - {y_key},
                state.remaining_edges,
            )
            for x_key in x.nodes.keys()
            if self.legal_node_mapping(x, y, state, x_key, y_key)
        ]

        if not next_states:
            return [
                State(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.remaining_nodes - {y_key},
                    state.remaining_edges,
                )
            ]

        return next_states

    def expand_edge(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        y_key: K,
    ) -> list[State[K]]:
        """Expand a given edge and its queue"""

        next_states: list[State[K]] = [
            State(
                state.mapped_nodes,
                state.mapped_edges.set(y_key, x_key),
                state.remaining_nodes,
                state.remaining_edges - {y_key},
            )
            for x_key in x.edges.keys()
            if self.legal_edge_mapping(x, y, state, x_key, y_key)
        ]

        if not next_states:
            next_states.append(
                State(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.remaining_nodes,
                    state.remaining_edges - {y_key},
                )
            )

        return next_states

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
        x_centralities: Mapping[K, float],
        y_centralities: Mapping[K, float],
    ) -> list[State[K]]:
        """Expand a given node and its queue"""

        selection = self.selection_func(
            x,
            y,
            state,
            node_pair_sims,
            edge_pair_sims,
            self.future_sim_func,
            x_centralities,
            y_centralities,
        )

        if selection is None:
            return []

        y_key, y_type = selection

        if y_type == "node":
            return self.expand_node(x, y, state, y_key)

        if y_type == "edge":
            return self.expand_edge(x, y, state, y_key)

        raise ValueError(f"Unknown element type: {y_type}")

    def compute_priority(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        past_sim = unpack_float(
            self.past_sim_func(x, y, state, node_pair_sims, edge_pair_sims)
        )
        future_sim = self.future_sim_func(x, y, state, node_pair_sims, edge_pair_sims)
        prio = 1 - (past_sim + future_sim)

        if self.pathlength_weight > 0:
            num_paths = len(state.mapped_nodes) + len(state.mapped_edges)
            return prio / (self.pathlength_weight**num_paths)

        return prio

    def __call_case_oriented__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        result = self(x=y, y=x)

        return GraphSim(
            result.value,
            {v: k for k, v in result.node_mapping.items()},
            {v: k for k, v in result.edge_mapping.items()},
            {result.node_mapping[k]: v for k, v in result.node_similarities.items()},
            {result.edge_mapping[k]: v for k, v in result.edge_similarities.items()},
        )

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform an A* analysis of the x base and the y"""
        if (
            (len(y.nodes) + len(y.edges)) > (len(x.nodes) + len(x.edges))
        ) and self.allow_case_oriented_mapping:
            return self.__call_case_oriented__(x, y)

        # TODO: compute node centrality scores using rustworkx
        x_centralities: dict[K, float] = {}
        y_centralities: dict[K, float] = {}

        node_pairs = [
            (x_node, y_node)
            for x_node, y_node in itertools.product(x.nodes.values(), y.nodes.values())
            if self.node_matcher(x_node.value, y_node.value)
        ]
        node_pair_sims = {
            (y_node.key, x_node.key): unpack_float(sim)
            for (x_node, y_node), sim in zip(
                node_pairs, self.batch_node_sim_func(node_pairs), strict=True
            )
        }

        edge_pairs = [
            (x_edge, y_edge)
            for x_edge, y_edge in itertools.product(x.edges.values(), y.edges.values())
            if self.edge_matcher(x_edge.value, y_edge.value)
            and self.node_matcher(
                x.nodes[x_edge.source.key].value,
                y.nodes[y_edge.source.key].value,
            )
            and self.node_matcher(
                x.nodes[x_edge.target.key].value,
                y.nodes[y_edge.target.key].value,
            )
        ]
        edge_pair_sims = {
            (y_edge.key, x_edge.key): unpack_float(sim)
            for (x_edge, y_edge), sim in zip(
                edge_pairs, self.batch_edge_sim_func(edge_pairs), strict=True
            )
        }

        open_set: list[PriorityState[K]] = []
        best_state = self.init_func(x, y, self.node_matcher, self.edge_matcher)
        heapq.heappush(open_set, PriorityState(0, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state

            if not current_state.remaining_nodes and not current_state.remaining_edges:
                best_state = current_state
                break

            next_states = self.expand(
                x,
                y,
                current_state,
                node_pair_sims,
                edge_pair_sims,
                x_centralities,
                y_centralities,
            )

            for next_state in next_states:
                next_prio = self.compute_priority(
                    x, y, next_state, node_pair_sims, edge_pair_sims
                )
                heapq.heappush(open_set, PriorityState(next_prio, next_state))

            if self.beam_width > 0 and len(open_set) > self.beam_width:
                open_set = open_set[: self.beam_width]

        sim = self.past_sim_func(x, y, best_state, node_pair_sims, edge_pair_sims)

        return GraphSim(
            unpack_float(sim),
            dict(best_state.mapped_nodes),
            dict(best_state.mapped_edges),
            dict(sim.node_similarities) if isinstance(sim, PastSim) else {},
            dict(sim.edge_similarities) if isinstance(sim, PastSim) else {},
        )
