import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from ...model.graph import (
    Edge,
    Graph,
    Node,
)
from ...typing import BatchSimFunc
from .common import BaseGraphSimFunc, SemanticEdgeSim


@dataclass(slots=True)
class precompute[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], BatchSimFunc[Graph[K, N, E, G], float]
):
    precompute_nodes: bool = True
    precompute_edges: bool = True

    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]]
    ) -> list[float]:
        if self.precompute_nodes:
            node_pairs: list[tuple[Node[K, N], Node[K, N]]] = []

            for x, y in batches:
                node_pairs.extend(
                    (x.nodes[x_key], y.nodes[y_key])
                    for x_key, y_key in itertools.product(
                        x.nodes.keys(), y.nodes.keys()
                    )
                    if self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value)
                )

            self.batch_node_sim_func(node_pairs)

        if self.precompute_edges and not isinstance(
            self.batch_edge_sim_func, SemanticEdgeSim
        ):
            edge_pairs: list[tuple[Edge[K, N, E], Edge[K, N, E]]] = []

            for x, y in batches:
                edge_pairs.extend(
                    (x.edges[x_key], y.edges[y_key])
                    for x_key, y_key in itertools.product(
                        x.edges.keys(), y.edges.keys()
                    )
                    if self.edge_matcher(x.edges[x_key].value, y.edges[y_key].value)
                    and self.node_matcher(
                        x.edges[x_key].source.value, y.edges[y_key].source.value
                    )
                    and self.node_matcher(
                        x.edges[x_key].target.value, y.edges[y_key].target.value
                    )
                )

            self.batch_edge_sim_func(edge_pairs)

        return [1.0] * len(batches)
