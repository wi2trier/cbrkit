import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from cbrkit.helpers import chain_map_chunks, unpack_float

from ...model.graph import (
    Graph,
    Node,
)
from ...typing import BatchSimFunc
from .common import BaseGraphSimFunc


@dataclass(slots=True)
class precompute[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], BatchSimFunc[Graph[K, N, E, G], float]
):
    precompute_nodes: bool = True
    precompute_edges: bool = True

    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]]
    ) -> list[float]:
        precompute_edges = (
            self.precompute_edges and self.edge_sim_func.edge_sim_func is not None
        )
        batch_node_pair_sims: list[dict[tuple[K, K], float]] = []

        if self.precompute_nodes or precompute_edges:
            batch_node_pairs: list[list[tuple[Node[K, N], Node[K, N]]]] = [
                [
                    (x_node, y_node)
                    for x_node, y_node in itertools.product(
                        x.nodes.values(), y.nodes.values()
                    )
                    if self.node_matcher(x_node.value, y_node.value)
                ]
                for x, y in batches
            ]

            batch_node_pair_sims_list = chain_map_chunks(
                batch_node_pairs, self.batch_node_sim_func
            )
            batch_node_pair_sims = [
                {
                    (y_node.key, x_node.key): unpack_float(sim)
                    for (x_node, y_node), sim in zip(
                        node_pair_values, node_pair_sims, strict=True
                    )
                }
                for node_pair_values, node_pair_sims in zip(
                    batch_node_pairs, batch_node_pair_sims_list, strict=True
                )
            ]

        if precompute_edges:
            edge_pairs: list[tuple[E, E, float, float]] = []

            for (x, y), node_pair_sims in zip(
                batches, batch_node_pair_sims, strict=True
            ):
                edge_pairs.extend(
                    (
                        x_edge.value,
                        y_edge.value,
                        node_pair_sims[(y_edge.source.key, x_edge.source.key)],
                        node_pair_sims[(y_edge.target.key, x_edge.target.key)],
                    )
                    for x_edge, y_edge in itertools.product(
                        x.edges.values(), y.edges.values()
                    )
                    if self.edge_matcher(x_edge.value, y_edge.value)
                    and (y_edge.source.key, x_edge.source.key) in node_pair_sims
                    and (y_edge.target.key, x_edge.target.key) in node_pair_sims
                )

            self.edge_sim_func(edge_pairs)

        return [1.0] * len(batches)
