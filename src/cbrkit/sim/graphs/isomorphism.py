from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

from ...helpers import batchify_sim
from ...typing import (
    AggregatorFunc,
    AnySimFunc,
    BatchSimFunc,
    Float,
    SimFunc,
)
from .model import Graph, GraphSim, Node


def default_edge_matcher[T](x: T, y: T) -> bool:
    return True


@dataclass(slots=True)
class isomorphism[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """Compute subgraph isomorphisms between two graphs.

    - Convert the input graphs to Rustworkx graphs.
    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the similarity based on the node mapping.
    - Return the isomorphism mapping with the highest similarity.
    """

    node_matcher: Callable[[N, N], bool]
    edge_matcher: Callable[[E, E], bool]
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    aggregator: AggregatorFunc[Any, Float]

    def __init__(
        self,
        node_matcher: Callable[[N, N], bool],
        aggregator: AggregatorFunc[Any, Float],
        node_sim_func: AnySimFunc[Node[K, N], Float],
        edge_matcher: Callable[[E, E], bool] | None = None,
    ) -> None:
        self.node_matcher = node_matcher
        self.edge_matcher = edge_matcher or default_edge_matcher
        self.aggregator = aggregator
        self.node_sim_func = batchify_sim(node_sim_func)

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        import rustworkx

        from .model import to_rustworkx_with_lookup

        x_rw, x_lookup = to_rustworkx_with_lookup(x)
        y_rw, y_lookup = to_rustworkx_with_lookup(y)

        rw_mappings = rustworkx.vf2_mapping(
            y_rw,
            x_rw,
            subgraph=True,
            node_matcher=self.node_matcher,
            edge_matcher=self.edge_matcher,
        )

        node_mappings: list[dict[K, K]] = [
            {y_lookup[y_key]: x_lookup[x_key] for y_key, x_key in mapping.items()}
            for mapping in rw_mappings
        ]

        if len(node_mappings) == 0:
            return GraphSim(0.0, node_mappings={}, edge_mappings={})

        mapping_similarities: list[float] = []

        for node_mapping in node_mappings:
            node_pairs = [
                (x.nodes[x_key], y.nodes[y_key])
                for y_key, x_key in node_mapping.items()
            ]
            node_similarities = self.node_sim_func(node_pairs)
            mapping_similarities.append(self.aggregator(node_similarities))

        best_mapping_id, best_sim = max(
            enumerate(mapping_similarities),
            key=lambda x: x[1],
        )
        best_mapping = node_mappings[best_mapping_id]

        return GraphSim(best_sim, node_mappings=best_mapping, edge_mappings={})
