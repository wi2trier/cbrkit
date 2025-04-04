import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, override

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
    Node,
)
from ...typing import AnySimFunc, BatchSimFunc, Float, SimFunc, StructuredValue
from ..wrappers import transpose_value
from .common import ElementMatcher, GraphSim, default_element_matcher

logger = get_logger(__name__)

__all__ = [
    "build",
    "init1",
    "default_edge_sim",
    "State",
    "StateSim",
    "InitFunc",
    "LegalMappingFunc",
    "build",
    "legal_node_mapping",
    "legal_edge_mapping",
]


@dataclass(slots=True, frozen=True)
class State[K]:
    # mappings are from y to x
    mapped_nodes: frozendict[K, K]
    mapped_edges: frozendict[K, K]
    open_node_pairs: frozenset[tuple[K, K]]
    open_edge_pairs: frozenset[tuple[K, K]]


@dataclass(slots=True, frozen=True)
class StateSim[K](StructuredValue[float]):
    node_similarities: Mapping[K, float]
    edge_similarities: Mapping[K, float]


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
class init1[K, N, E, G](InitFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> State[K]:
        return State(
            frozendict(),
            frozendict(),
            frozenset(itertools.product(y.nodes.keys(), x.nodes.keys())),
            frozenset(itertools.product(y.edges.keys(), x.edges.keys())),
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


@dataclass(slots=True, init=False)
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

    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]
    init_func: InitFunc[K, N, E, G]
    legal_node_mapping: LegalMappingFunc[K, N, E, G]
    legal_edge_mapping: LegalMappingFunc[K, N, E, G]
    start_with: Literal["nodes", "edges"]

    def __init__(
        self,
        node_sim_func: AnySimFunc[N, Float],
        node_matcher: ElementMatcher[N],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
        edge_matcher: ElementMatcher[E] = default_element_matcher,
        init_func: InitFunc[K, N, E, G] | None = None,
        start_with: Literal["nodes", "edges"] = "nodes",
    ) -> None:
        self.legal_node_mapping = legal_node_mapping(node_matcher)
        self.legal_edge_mapping = legal_edge_mapping(edge_matcher)

        self.node_sim_func = batchify_sim(transpose_value(node_sim_func))
        self.edge_sim_func = (
            default_edge_sim(self.node_sim_func)
            if edge_sim_func is None
            else batchify_sim(edge_sim_func)
        )
        self.init_func = init_func if init_func else init1()

        self.start_with = start_with

    def compute_similarity(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> StateSim[K]:
        """Function to compute the similarity based on the current state"""

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

        return StateSim(
            sum(all_sims) / total_elements,
            dict(zip(s.mapped_nodes.keys(), node_sims, strict=True)),
            dict(zip(s.mapped_edges.keys(), edge_sims, strict=True)),
        )

    def expand_edges(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
    ) -> list[State[K]]:
        """Expand the current state by adding all possible edge mappings"""

        new_states = []

        for y_key, x_key in state.open_edge_pairs:
            if self.legal_edge_mapping(x, y, state, x_key, y_key):
                new_state = State(
                    state.mapped_nodes,
                    state.mapped_edges.set(y_key, x_key),
                    state.open_node_pairs,
                    state.open_edge_pairs
                    - {
                        (y, x)
                        for y, x in itertools.product(y.edges.keys(), x.edges.keys())
                        if y == y_key or x == x_key
                    },
                )
                new_states.append(new_state)

        return new_states

    def expand_nodes(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
    ) -> list[State[K]]:
        """Expand the current state by adding all possible node mappings"""

        new_states = []

        for y_key, x_key in state.open_node_pairs:
            if self.legal_node_mapping(x, y, state, x_key, y_key):
                new_state = State(
                    state.mapped_nodes.set(y_key, x_key),
                    state.mapped_edges,
                    state.open_node_pairs
                    - {
                        (y, x)
                        for y, x in itertools.product(y.nodes.keys(), x.nodes.keys())
                        if y == y_key or x == x_key
                    },
                    state.open_edge_pairs,
                )
                new_states.append(new_state)

        return new_states

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        current_state: State[K],
        current_sim: StateSim[K],
        expand_func: Callable[
            [Graph[K, N, E, G], Graph[K, N, E, G], State[K]], list[State[K]]
        ],
    ) -> tuple[State[K], StateSim[K]]:
        """Expand the current state by adding all possible mappings"""

        while True:
            # Iterate over all open pairs and find the best pair
            new_states = expand_func(x, y, current_state)
            new_sims = [self.compute_similarity(x, y, state) for state in new_states]

            best_sim, best_state = max(
                zip(new_sims, new_states, strict=True), key=lambda item: item[0].value
            )

            # If no better pair is found, break the loop
            if best_state == current_state and best_sim == current_sim:
                break

            # Update the current state and similarity
            current_state = best_state
            current_sim = best_sim

        return current_state, current_sim

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform greedy graph matching of the query y against the case x"""

        current_state = self.init_func(x, y)
        current_sim = self.compute_similarity(x, y, current_state)

        if self.start_with == "edges":
            current_state, current_sim = self.expand(
                x, y, current_state, current_sim, self.expand_edges
            )
            current_state, current_sim = self.expand(
                x, y, current_state, current_sim, self.expand_nodes
            )
        elif self.start_with == "nodes":
            current_state, current_sim = self.expand(
                x, y, current_state, current_sim, self.expand_nodes
            )
            current_state, current_sim = self.expand(
                x, y, current_state, current_sim, self.expand_edges
            )
        else:
            raise ValueError(
                f"Invalid start_with value: {self.start_with}. Expected 'nodes' or 'edges'."
            )

        return GraphSim(
            current_sim.value,
            dict(current_state.mapped_nodes),
            dict(current_state.mapped_edges),
            dict(current_sim.node_similarities)
            if isinstance(current_sim, StateSim)
            else {},
            dict(current_sim.edge_similarities)
            if isinstance(current_sim, StateSim)
            else {},
        )
