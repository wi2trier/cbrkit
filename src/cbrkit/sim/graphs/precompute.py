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
                    (x_node, y_node)
                    for x_node, y_node in itertools.product(
                        x.nodes.values(), y.nodes.values()
                    )
                    if self.node_matcher(x_node.value, y_node.value)
                )

            self.batch_node_sim_func(node_pairs)

        if self.precompute_edges and not isinstance(
            self.batch_edge_sim_func, SemanticEdgeSim
        ):
            edge_pairs: list[tuple[Edge[K, N, E], Edge[K, N, E]]] = []

            for x, y in batches:
                edge_pairs.extend(
                    (x_edge, y_edge)
                    for x_edge, y_edge in itertools.product(
                        x.edges.values(), y.edges.values()
                    )
                    if self.edge_matcher(x_edge.value, y_edge.value)
                    and self.node_matcher(x_edge.source.value, y_edge.source.value)
                    and self.node_matcher(x_edge.target.value, y_edge.target.value)
                )

            self.batch_edge_sim_func(edge_pairs)

        return [1.0] * len(batches)
