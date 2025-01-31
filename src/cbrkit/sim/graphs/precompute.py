import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from ...helpers import (
    batchify_sim,
)
from ...model.graph import (
    Edge,
    Graph,
    Node,
)
from ...typing import AnySimFunc, BatchSimFunc, Float


@dataclass(slots=True, frozen=True)
class precompute[K, N, E, G](BatchSimFunc[Graph[K, N, E, G], float]):
    nodes_sim_func: AnySimFunc[Node[K, N], Float] | None = None
    edges_sim_sumc: AnySimFunc[Edge[K, N, E], Float] | None = None

    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]]
    ) -> list[float]:
        if self.nodes_sim_func is not None:
            node_sim_func = batchify_sim(self.nodes_sim_func)
            node_batches: list[tuple[Node[K, N], Node[K, N]]] = []

            for x, y in batches:
                node_batches.extend(
                    [
                        (x.nodes[x_key], y.nodes[y_key])
                        for x_key, y_key in itertools.product(
                            x.nodes.keys(), y.nodes.keys()
                        )
                    ]
                )

            node_sim_func(node_batches)

        if self.edges_sim_sumc is not None:
            edge_sim_func = batchify_sim(self.edges_sim_sumc)
            edge_batches: list[tuple[Edge[K, N, E], Edge[K, N, E]]] = []

            for x, y in batches:
                edge_batches.extend(
                    [
                        (x.edges[x_key], y.edges[y_key])
                        for x_key, y_key in itertools.product(
                            x.edges.keys(), y.edges.keys()
                        )
                    ]
                )

            edge_sim_func(edge_batches)

        return [1.0] * len(batches)
