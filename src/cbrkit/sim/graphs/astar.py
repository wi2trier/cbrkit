import heapq
import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Protocol, override

from frozendict import frozendict

from ...helpers import (
    batchify_sim,
    get_logger,
    unpack_float,
    unpack_floats,
)
from ...model.graph import (
    Edge,
    Graph,
    GraphElementType,
    Node,
)
from ...typing import AnySimFunc, BatchSimFunc, Float, SimFunc, StructuredValue
from ..wrappers import transpose_value
from .common import ElementMatcher, GraphSim, default_element_matcher

__all__ = [
    "PastSimFunc",
    "FutureSimFunc",
    "SelectionFunc",
    "InitFunc",
    "LegalMappingFunc",
    "h1",
    "h2",
    "h3",
    "g1",
    "select1",
    "select2",
    "select3",
    "init1",
    "init2",
    "legal_node_mapping",
    "legal_edge_mapping",
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
        /,
    ) -> float: ...


class PastSimFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        /,
    ) -> float | PastSim[K]: ...


class SelectionFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        /,
    ) -> None | tuple[K, GraphElementType]: ...


class InitFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        /,
    ) -> State[K]: ...


class LegalMappingFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        x_key: K,
        y_key: K,
    ) -> bool: ...


@dataclass(slots=True, frozen=True)
class h1[K, N, E, G](FutureSimFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        """Heuristic to compute future costs"""

        return (len(s.remaining_nodes) + len(s.remaining_edges)) / (
            len(y.nodes) + len(y.edges)
        )


@dataclass(slots=True)
class h2[K, N, E, G](FutureSimFunc[K, N, E, G]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[N, Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(transpose_value(node_sim_func))
        self.edge_sim_func = (
            default_edge_sim(self.node_sim_func)
            if edge_sim_func is None
            else batchify_sim(edge_sim_func)
        )

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        h_val = 0

        for y_name in s.remaining_nodes:
            max_sim = max(
                unpack_floats(
                    self.node_sim_func(
                        [(x_node, y.nodes[y_name]) for x_node in x.nodes.values()]
                    )
                ),
                default=0,
            )

            h_val += max_sim

        for y_name in s.remaining_edges:
            max_sim = max(
                unpack_floats(
                    self.edge_sim_func(
                        [(x_edge, y.edges[y_name]) for x_edge in x.edges.values()]
                    )
                ),
                default=0,
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class h3[K, N, E, G](FutureSimFunc[K, N, E, G]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[N, Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(transpose_value(node_sim_func))
        self.edge_sim_func = (
            default_edge_sim(self.node_sim_func)
            if edge_sim_func is None
            else batchify_sim(edge_sim_func)
        )

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        h_val = 0

        for y_name in s.remaining_nodes:
            y_node = y.nodes[y_name]
            max_sim = max(
                unpack_floats(
                    self.node_sim_func(
                        [
                            (x_node, y_node)
                            for x_name, x_node in x.nodes.items()
                            if x_name not in s.mapped_nodes.values()
                        ]
                    )
                ),
                default=0,
            )

            h_val += max_sim

        for y_name in s.remaining_edges:
            y_edge = y.edges[y_name]
            max_sim = max(
                unpack_floats(
                    self.edge_sim_func(
                        [
                            (x_edge, y_edge)
                            for x_name, x_edge in x.edges.items()
                            if x_name not in s.mapped_edges.values()
                            and y_edge.source.key not in s.mapped_nodes
                            and y_edge.target.key not in s.mapped_nodes
                            and x_edge.source.key not in s.mapped_nodes.values()
                            and x_edge.target.key not in s.mapped_nodes.values()
                        ]
                    )
                ),
                default=0,
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


@dataclass(slots=True)
class g1[K, N, E, G](PastSimFunc[K, N, E, G]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[N, Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(transpose_value(node_sim_func))
        self.edge_sim_func = (
            default_edge_sim(self.node_sim_func)
            if edge_sim_func is None
            else batchify_sim(edge_sim_func)
        )

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> PastSim[K]:
        """Function to compute the costs of all previous steps"""

        node_sims = unpack_floats(
            self.node_sim_func(
                [
                    (x.nodes[x_key], y.nodes[y_key])
                    for y_key, x_key in s.mapped_nodes.items()
                ]
            )
        )

        edge_sims = unpack_floats(
            self.edge_sim_func(
                [
                    (x.edges[x_key], y.edges[y_key])
                    for y_key, x_key in s.mapped_edges.items()
                ]
            )
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
    heuristic_func: FutureSimFunc[K, N, E, G]

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> None | tuple[K, GraphElementType]:
        """Select the next node or edge to be mapped"""

        heuristic_scores: list[tuple[K, GraphElementType, float]] = []

        for y_key in s.remaining_nodes:
            heuristic_scores.append((y_key, "node", self.heuristic_func(x, y, s)))

        for y_key in s.remaining_edges:
            heuristic_scores.append((y_key, "edge", self.heuristic_func(x, y, s)))

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
    ) -> State[K]:
        return State(
            frozendict(),
            frozendict(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
        )


@dataclass(slots=True, init=False)
class init2[K, N, E, G](InitFunc[K, N, E, G]):
    legal_node_mapping: LegalMappingFunc[K, N, E, G]
    legal_edge_mapping: LegalMappingFunc[K, N, E, G]

    def __init__(
        self,
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E] = default_element_matcher,
    ):
        self.legal_node_mapping = legal_node_mapping(node_matcher)
        self.legal_edge_mapping = legal_edge_mapping(edge_matcher)

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> State[K]:
        # pre-populate the mapping with nodes/edges that only have one possible legal mapping
        possible_node_mappings: defaultdict[K, set[K]] = defaultdict(set)
        possible_edge_mappings: defaultdict[K, set[K]] = defaultdict(set)

        state = State(
            frozendict(),
            frozendict(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
        )

        for y_key, x_key in itertools.product(y.nodes.keys(), x.nodes.keys()):
            if self.legal_node_mapping(x, y, state, x_key, y_key):
                possible_node_mappings[y_key].add(x_key)

        for y_key, x_key in itertools.product(y.edges.keys(), x.edges.keys()):
            if self.legal_edge_mapping(x, y, state, x_key, y_key):
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


@dataclass(slots=True, frozen=True)
class default_edge_sim[K, N, E](BatchSimFunc[Edge[K, N, E], Float]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]

    @override
    def __call__(
        self, batches: Sequence[tuple[Edge[K, N, E], Edge[K, N, E]]]
    ) -> list[float]:
        source_sims = self.node_sim_func([(x.source, y.source) for x, y in batches])
        target_sims = self.node_sim_func([(x.target, y.target) for x, y in batches])

        return [
            0.5 * (unpack_float(source) + unpack_float(target))
            for source, target in zip(source_sims, target_sims, strict=True)
        ]


@dataclass(slots=True, frozen=True)
class legal_node_mapping[K, N, E, G](LegalMappingFunc[K, N, E, G]):
    matcher: ElementMatcher[N]

    def __call__(
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
            and self.matcher(x.nodes[x_key].value, y.nodes[y_key].value)
        )


@dataclass(slots=True, frozen=True)
class legal_edge_mapping[K, N, E, G](LegalMappingFunc[K, N, E, G]):
    matcher: ElementMatcher[E]

    def __call__(
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
            and self.matcher(x_value.value, y_value.value)
            # if the nodes are already mapped, check if they are mapped legally
            and (
                mapped_x_source_key is None or x_value.source.key == mapped_x_source_key
            )
            and (
                mapped_x_target_key is None or x_value.target.key == mapped_x_target_key
            )
        )


@dataclass(slots=True)
class build[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        past_cost_func: A heuristic function to compute the costs of all previous steps.
        future_cost_func: A heuristic function to compute the future costs.
        selection_func: A function to select the next node or edge to be mapped.
        init_func: A function to initialize the state.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    past_cost_func: PastSimFunc[K, N, E, G]
    future_cost_func: FutureSimFunc[K, N, E, G]
    node_matcher: InitVar[ElementMatcher[N]]
    edge_matcher: InitVar[ElementMatcher[E]] = default_element_matcher
    selection_func: SelectionFunc[K, N, E, G] = field(default_factory=select2)
    init_func: InitFunc[K, N, E, G] = field(default_factory=init1)
    queue_limit: int = 10000
    legal_node_mapping: LegalMappingFunc[K, N, E, G] = field(init=False)
    legal_edge_mapping: LegalMappingFunc[K, N, E, G] = field(init=False)
    # TODO: Currently not implemented as described in the paper, needs further investigation
    allow_case_oriented_mapping: bool = False

    def __post_init__(
        self, node_matcher: ElementMatcher[N], edge_matcher: ElementMatcher[E]
    ) -> None:
        self.legal_node_mapping = legal_node_mapping(node_matcher)
        self.legal_edge_mapping = legal_edge_mapping(edge_matcher)

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
    ) -> list[State[K]]:
        """Expand a given node and its queue"""

        selection = self.selection_func(x, y, state)

        if selection is None:
            return []

        y_key, y_type = selection

        if y_type == "node":
            return self.expand_node(x, y, state, y_key)

        if y_type == "edge":
            return self.expand_edge(x, y, state, y_key)

        raise ValueError(f"Unknown element type: {y_type}")

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

        open_set: list[PriorityState[K]] = []
        best_state = self.init_func(x, y)
        heapq.heappush(open_set, PriorityState(0, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_state = first_elem.state

            if not current_state.remaining_nodes and not current_state.remaining_edges:
                best_state = current_state
                break

            for next_state in self.expand(x, y, current_state):
                next_sim = unpack_float(
                    self.past_cost_func(x, y, next_state)
                ) + self.future_cost_func(x, y, next_state)
                heapq.heappush(open_set, PriorityState(1 - next_sim, next_state))

            if self.queue_limit > 0 and len(open_set) > self.queue_limit:
                open_set = open_set[: self.queue_limit]

        sim = self.past_cost_func(x, y, best_state)

        return GraphSim(
            unpack_float(sim),
            dict(best_state.mapped_nodes),
            dict(best_state.mapped_edges),
            dict(sim.node_similarities) if isinstance(sim, PastSim) else {},
            dict(sim.edge_similarities) if isinstance(sim, PastSim) else {},
        )
