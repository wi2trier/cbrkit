import itertools
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from typing import Any, Protocol

from frozendict import frozendict

from ...helpers import batchify_sim, unpack_float
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


def type_element_matcher(x: Any, y: Any) -> bool:
    return type(x) is type(y)


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
    node_sim_func: InitVar[AnySimFunc[N, Float]]
    node_matcher: ElementMatcher[N]
    edge_sim_func: InitVar[
        AnySimFunc[Edge[K, N, E], Float] | SemanticEdgeSim[K, N, E] | None
    ] = None
    edge_matcher: ElementMatcher[E] = default_element_matcher
    batch_node_sim_func: BatchSimFunc[Node[K, N], Float] = field(init=False)
    batch_edge_sim_func: (
        BatchSimFunc[Edge[K, N, E], Float] | SemanticEdgeSim[K, N, E]
    ) = field(init=False)

    def __post_init__(
        self,
        any_node_sim_func: AnySimFunc[N, Float],
        any_edge_sim_func: AnySimFunc[Edge[K, N, E], Float]
        | SemanticEdgeSim[K, N, E]
        | None,
    ) -> None:
        self.batch_node_sim_func = batchify_sim(transpose_value(any_node_sim_func))

        if isinstance(any_edge_sim_func, SemanticEdgeSim):
            self.batch_edge_sim_func = any_edge_sim_func
        elif any_edge_sim_func is None:
            self.batch_edge_sim_func = SemanticEdgeSim()
        else:
            self.batch_edge_sim_func = batchify_sim(any_edge_sim_func)

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


@dataclass(slots=True, frozen=True)
class SearchState[K]:
    # mappings are from y to x
    mapped_nodes: frozendict[K, K]
    mapped_edges: frozendict[K, K]
    open_nodes: frozenset[K]
    open_edges: frozenset[K]


class SearchGraphSimFunc[K, N, E, G](BaseGraphSimFunc[K, N, E, G]):
    def legal_node_mapping(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
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
        state: SearchState[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        x_value = x.edges[x_key]
        y_value = y.edges[y_key]

        return (
            y_key not in state.mapped_edges.keys()
            and x_key not in state.mapped_edges.values()
            and self.edge_matcher(x_value.value, y_value.value)
            # source and target of the edge must be mapped to the same nodes
            and x_value.source.key == state.mapped_nodes.get(y_value.source.key)
            and x_value.target.key == state.mapped_nodes.get(y_value.target.key)
        )

    def expand_node(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:
        """Expand a given node and its queue"""

        next_states: list[SearchState[K]] = [
            SearchState(
                state.mapped_nodes.set(y_key, x_key),
                state.mapped_edges,
                state.open_nodes - {y_key},
                state.open_edges,
            )
            for x_key in x.nodes.keys()
            if self.legal_node_mapping(x, y, state, x_key, y_key)
        ]

        if not next_states:
            return [
                SearchState(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.open_nodes - {y_key},
                    state.open_edges,
                )
            ]

        return next_states

    def expand_edge(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:
        """Expand a given edge and its queue"""

        next_states: list[SearchState[K]] = []

        for x_key in x.edges.keys():
            next_state = state
            x_source_key = x.edges[x_key].source.key
            x_target_key = x.edges[x_key].target.key
            y_source_key = y.edges[y_key].source.key
            y_target_key = y.edges[y_key].target.key

            if (
                y_source_key not in next_state.mapped_nodes.keys()
                and x_source_key not in next_state.mapped_nodes.values()
                and self.legal_node_mapping(
                    x, y, next_state, x_source_key, y_source_key
                )
            ):
                next_state = SearchState(
                    next_state.mapped_nodes.set(y_source_key, x_source_key),
                    next_state.mapped_edges,
                    next_state.open_nodes - {y_source_key},
                    next_state.open_edges,
                )

            if (
                y_target_key not in next_state.mapped_nodes.keys()
                and x_target_key not in next_state.mapped_nodes.values()
                and self.legal_node_mapping(
                    x, y, next_state, x_target_key, y_target_key
                )
            ):
                next_state = SearchState(
                    next_state.mapped_nodes.set(y_target_key, x_target_key),
                    next_state.mapped_edges,
                    next_state.open_nodes - {y_target_key},
                    next_state.open_edges,
                )

            if self.legal_edge_mapping(x, y, next_state, x_key, y_key):
                next_states.append(
                    SearchState(
                        next_state.mapped_nodes,
                        next_state.mapped_edges.set(y_key, x_key),
                        next_state.open_nodes,
                        next_state.open_edges - {y_key},
                    )
                )

        if not next_states:
            next_states.append(
                SearchState(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.open_nodes,
                    state.open_edges - {y_key},
                )
            )

        return next_states
