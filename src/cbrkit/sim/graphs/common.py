import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from frozendict import frozendict

from ...helpers import (
    batchify_sim,
    reverse_batch_positional,
    reverse_positional,
    unpack_float,
)
from ...model.graph import Edge, Graph, Node
from ...typing import AnySimFunc, BatchSimFunc, Float, StructuredValue
from ..wrappers import transpose_value

type PairSim[K] = Mapping[tuple[K, K], float]


@dataclass(slots=True, frozen=True)
class GraphSim[K](StructuredValue[float]):
    node_mapping: frozendict[K, K]
    edge_mapping: frozendict[K, K]
    node_similarities: frozendict[K, float]
    edge_similarities: frozendict[K, float]


class ElementMatcher[T](Protocol):
    def __call__(self, x: T, y: T, /) -> bool: ...


def default_element_matcher(x: Any, y: Any) -> bool:
    return True


@dataclass(slots=True, frozen=True)
class SemanticEdgeSim[K, N, E]:
    source_weight: float = 0.5
    target_weight: float = 0.5

    def __call__(
        self,
        batches: Sequence[tuple[Edge[K, N, E], Edge[K, N, E], PairSim[K]]],
    ) -> list[float]:
        source_sims = (
            node_pair_sims.get((y.source.key, x.source.key), 0.0)
            for x, y, node_pair_sims in batches
        )
        target_sims = (
            node_pair_sims.get((y.target.key, x.target.key), 0.0)
            for x, y, node_pair_sims in batches
        )

        return [
            (self.source_weight * source + self.target_weight * target)
            / (self.source_weight + self.target_weight)
            for source, target in zip(source_sims, target_sims, strict=True)
        ]


@dataclass(slots=True)
class BaseGraphSimFunc[K, N, E, G]:
    node_sim_func: AnySimFunc[N, Float]
    edge_sim_func: (
        AnySimFunc[Edge[K, N, E], Float] | SemanticEdgeSim[K, N, E] | None
    ) = None
    node_matcher: ElementMatcher[N] = default_element_matcher
    edge_matcher: ElementMatcher[E] = default_element_matcher
    batch_node_sim_func: BatchSimFunc[Node[K, N], Float] = field(init=False)
    batch_edge_sim_func: (
        BatchSimFunc[Edge[K, N, E], Float] | SemanticEdgeSim[K, N, E]
    ) = field(init=False)
    _invert: bool = False

    def __post_init__(self) -> None:
        self.batch_node_sim_func = batchify_sim(transpose_value(self.node_sim_func))

        if isinstance(self.edge_sim_func, SemanticEdgeSim):
            self.batch_edge_sim_func = self.edge_sim_func
        elif self.edge_sim_func is None:
            self.batch_edge_sim_func = SemanticEdgeSim()
        else:
            self.batch_edge_sim_func = batchify_sim(self.edge_sim_func)

        if self._invert:
            self.node_matcher = reverse_positional(self.node_matcher)
            self.edge_matcher = reverse_positional(self.edge_matcher)
            self.batch_node_sim_func = reverse_batch_positional(
                self.batch_node_sim_func
            )
            if not isinstance(self.batch_edge_sim_func, SemanticEdgeSim):
                # semantic edge sim is agnostic to order
                self.batch_edge_sim_func = reverse_batch_positional(
                    self.batch_edge_sim_func
                )

    def induced_edge_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_mapping: Mapping[K, K],
    ) -> frozendict[K, K]:
        return frozendict(
            (y_value.key, x_value.key)
            for y_value, x_value in itertools.product(
                y.edges.values(), x.edges.values()
            )
            if self.edge_matcher(x_value.value, y_value.value)
            and x_value.source.key == node_mapping.get(y_value.source.key)
            and x_value.target.key == node_mapping.get(y_value.target.key)
        )

    def node_pair_similarities(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        pairs: Sequence[tuple[K, K]] | None = None,
    ) -> PairSim[K]:
        if pairs is None:
            pairs = [
                (y_key, x_key)
                for x_key, y_key in itertools.product(x.nodes.keys(), y.nodes.keys())
                if self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value)
            ]

        node_pair_values = [(x.nodes[x_key], y.nodes[y_key]) for y_key, x_key in pairs]
        node_pair_sims = self.batch_node_sim_func(node_pair_values)

        return {
            (y_node.key, x_node.key): unpack_float(sim)
            for (x_node, y_node), sim in zip(
                node_pair_values, node_pair_sims, strict=True
            )
        }

    def edge_pair_similarities(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_pair_sims: PairSim[K],
        pairs: Sequence[tuple[K, K]] | None = None,
    ) -> PairSim[K]:
        if pairs is None:
            pairs = [
                (y_key, x_key)
                for x_key, y_key in itertools.product(x.edges.keys(), y.edges.keys())
                if (y.edges[y_key].source.key, x.edges[x_key].source.key)
                in node_pair_sims
                and (y.edges[y_key].target.key, x.edges[x_key].target.key)
                in node_pair_sims
                and self.edge_matcher(x.edges[x_key].value, y.edges[y_key].value)
            ]

        edge_pair_values = [(x.edges[x_key], y.edges[y_key]) for y_key, x_key in pairs]

        if isinstance(self.batch_edge_sim_func, SemanticEdgeSim):
            edge_pair_sims = self.batch_edge_sim_func(
                [
                    (x_edge, y_edge, node_pair_sims)
                    for x_edge, y_edge in edge_pair_values
                ]
            )
        else:
            edge_pair_sims = self.batch_edge_sim_func(edge_pair_values)

        return {
            (y_edge.key, x_edge.key): unpack_float(sim)
            for (x_edge, y_edge), sim in zip(
                edge_pair_values, edge_pair_sims, strict=True
            )
        }

    def pair_similarities(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_pairs: Sequence[tuple[K, K]] | None = None,
        edge_pairs: Sequence[tuple[K, K]] | None = None,
    ) -> tuple[PairSim[K], PairSim[K]]:
        node_pair_sims = self.node_pair_similarities(x, y, node_pairs)
        edge_pair_sims = self.edge_pair_similarities(x, y, node_pair_sims, edge_pairs)
        return node_pair_sims, edge_pair_sims

    def similarity(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        mapped_nodes: frozendict[K, K],
        mapped_edges: frozendict[K, K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> GraphSim[K]:
        """Function to compute the similarity of all previous steps"""

        node_sims = [
            node_pair_sims[(y_key, x_key)] for y_key, x_key in mapped_nodes.items()
        ]

        edge_sims = [
            edge_pair_sims[(y_key, x_key)] for y_key, x_key in mapped_edges.items()
        ]

        all_sims = itertools.chain(node_sims, edge_sims)
        total_elements = len(y.nodes) + len(y.edges)
        total_sim = sum(all_sims) / total_elements if total_elements > 0 else 0.0

        return GraphSim(
            total_sim,
            mapped_nodes,
            mapped_edges,
            frozendict(zip(mapped_nodes.keys(), node_sims, strict=True)),
            frozendict(zip(mapped_edges.keys(), edge_sims, strict=True)),
        )

    def invert_similarity(
        self, x: Graph[K, N, E, G], y: Graph[K, N, E, G], sim: GraphSim[K]
    ) -> GraphSim[K]:
        node_mapping = frozendict((v, k) for k, v in sim.node_mapping.items())
        edge_mapping = frozendict((v, k) for k, v in sim.edge_mapping.items())

        node_similarities, edge_similarities = self.pair_similarities(
            x, y, list(node_mapping.items()), list(edge_mapping.items())
        )

        return self.similarity(
            x,
            y,
            node_mapping,
            edge_mapping,
            frozendict(node_similarities),
            frozendict(edge_similarities),
        )


@dataclass(slots=True, frozen=True)
class SearchState[K]:
    # mappings are from y/query to x/case
    node_mapping: frozendict[K, K]
    edge_mapping: frozendict[K, K]
    # contains all elements from the query that are not yet mapped
    # can be different from mapping.keys() if no candidate in x/case exists
    open_y_nodes: frozenset[K]
    open_y_edges: frozenset[K]
    # contains all elements from the case that are not yet mapped
    # must be identical to mapping.values() but is stored to optimize lookup
    open_x_nodes: frozenset[K]
    open_x_edges: frozenset[K]


class SearchGraphSimFunc[K, N, E, G](BaseGraphSimFunc[K, N, E, G]):
    def finished(self, state: SearchState[K]) -> bool:
        # the following condition could save a few iterations, but needs to be tested
        # return (not state.open_y_nodes and not state.open_y_edges) or (
        #     not state.open_x_nodes and not state.open_x_edges
        # )
        return not state.open_y_nodes and not state.open_y_edges

    def legal_node_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        return (
            self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value)
            and y_key in state.open_y_nodes
            and x_key in state.open_x_nodes
        )

    def legal_edge_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        x_value = x.edges[x_key]
        y_value = y.edges[y_key]

        return (
            self.edge_matcher(x_value.value, y_value.value)
            and y_key in state.open_y_edges
            and x_key in state.open_x_edges
            # source and target of the edge must be mapped to the same nodes
            and x_value.source.key == state.node_mapping.get(y_value.source.key)
            and x_value.target.key == state.node_mapping.get(y_value.target.key)
        )

    def expand_node(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:
        next_states: list[SearchState[K]] = [
            SearchState(
                state.node_mapping.set(y_key, x_key),
                state.edge_mapping,
                state.open_y_nodes - {y_key},
                state.open_y_edges,
                state.open_x_nodes - {x_key},
                state.open_x_edges,
            )
            for x_key in state.open_x_nodes
            if self.legal_node_mapping(x, y, state, x_key, y_key)
        ]

        if not next_states:
            next_states.append(
                SearchState(
                    state.node_mapping,
                    state.edge_mapping,
                    state.open_y_nodes - {y_key},
                    state.open_y_edges,
                    state.open_x_nodes,
                    state.open_x_edges,
                )
            )

        return next_states

    def expand_edge(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:

        next_states: list[SearchState[K]] = []

        for x_key in state.open_x_edges:
            next_state = state
            x_source_key = x.edges[x_key].source.key
            x_target_key = x.edges[x_key].target.key
            y_source_key = y.edges[y_key].source.key
            y_target_key = y.edges[y_key].target.key

            if (
                y_source_key in next_state.open_y_nodes
                and x_source_key in next_state.open_x_nodes
                and self.legal_node_mapping(
                    x, y, next_state, x_source_key, y_source_key
                )
            ):
                next_state = SearchState(
                    next_state.node_mapping.set(y_source_key, x_source_key),
                    next_state.edge_mapping,
                    next_state.open_y_nodes - {y_source_key},
                    next_state.open_y_edges,
                    next_state.open_x_nodes - {x_source_key},
                    next_state.open_x_edges,
                )

            if (
                y_target_key in next_state.open_y_nodes
                and x_target_key in next_state.open_x_nodes
                and self.legal_node_mapping(
                    x, y, next_state, x_target_key, y_target_key
                )
            ):
                next_state = SearchState(
                    next_state.node_mapping.set(y_target_key, x_target_key),
                    next_state.edge_mapping,
                    next_state.open_y_nodes - {y_target_key},
                    next_state.open_y_edges,
                    next_state.open_x_nodes - {x_target_key},
                    next_state.open_x_edges,
                )

            if self.legal_edge_mapping(x, y, next_state, x_key, y_key):
                next_states.append(
                    SearchState(
                        next_state.node_mapping,
                        next_state.edge_mapping.set(y_key, x_key),
                        next_state.open_y_nodes,
                        next_state.open_y_edges - {y_key},
                        next_state.open_x_nodes,
                        next_state.open_x_edges - {x_key},
                    )
                )

        if not next_states:
            next_states.append(
                SearchState(
                    state.node_mapping,
                    state.edge_mapping,
                    state.open_y_nodes,
                    state.open_y_edges - {y_key},
                    state.open_x_nodes,
                    state.open_x_edges,
                )
            )

        return next_states
