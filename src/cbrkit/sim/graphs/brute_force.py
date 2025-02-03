import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from ...helpers import batchify_sim, unpack_float, unpack_floats
from ...model.graph import (
    Edge,
    Graph,
    Node,
)
from ...typing import (
    AnySimFunc,
    BatchSimFunc,
    Float,
    SimFunc,
)
from ..wrappers import transpose_value
from .common import GraphSim


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


@dataclass
class brute_force[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Computes the similarity between two graphs by exhaustively computing all possible mappings
    and selecting the one with the highest similarity score.

    Args:
        node_sim_func: A similarity function for graph nodes that receives two node objects.
        edge_sim_func: A similarity function for graph edges that receives two edge objects.

    Returns:
        The similarity between the two graphs and the best mapping.
    """

    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[N, Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(transpose_value(node_sim_func))
        if edge_sim_func is None:
            self.edge_sim_func = default_edge_sim(self.node_sim_func)
        else:
            self.edge_sim_func = batchify_sim(edge_sim_func)

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        best_sim = 0.0
        best_node_mapping: dict[K, K] = {}
        best_edge_mapping: dict[K, K] = {}

        x_nodes = list(x.nodes.values())
        y_nodes = list(y.nodes.values())

        x_node_keys = [node.key for node in x_nodes]
        y_node_keys = [node.key for node in y_nodes]

        # Only consider mappings when y has equal or fewer nodes than x
        if len(y_nodes) > len(x_nodes):
            raise ValueError("Graph y has more nodes than graph x")

        # Generate all possible injective mappings from y_nodes to subsets of x_nodes
        x_node_combinations = itertools.combinations(x_node_keys, len(y_nodes))

        for x_combination in x_node_combinations:
            x_permutations = itertools.permutations(x_combination)

            for x_perm in x_permutations:
                node_mapping = dict(zip(y_node_keys, x_perm, strict=False))

                # Compute node similarities
                node_pairs = [
                    (y.nodes[y_key], x.nodes[x_key])
                    for y_key, x_key in node_mapping.items()
                ]
                node_sims = self.node_sim_func(node_pairs)
                total_node_sim = (
                    sum(unpack_floats(node_sims)) / len(node_sims) if node_sims else 0.0
                )

                # Build edge mappings based on node mappings
                edge_pairs = []

                for y_edge in y.edges.values():
                    y_source_key = y_edge.source.key
                    y_target_key = y_edge.target.key

                    if y_source_key in node_mapping and y_target_key in node_mapping:
                        x_source_key = node_mapping[y_source_key]
                        x_target_key = node_mapping[y_target_key]
                        # Find corresponding edge in x
                        x_edge = next(
                            (
                                e
                                for e in x.edges.values()
                                if e.source.key == x_source_key
                                and e.target.key == x_target_key
                            ),
                            None,
                        )

                        if x_edge is not None:
                            edge_pairs.append((y_edge, x_edge))

                # Compute edge similarities
                edge_sims = self.edge_sim_func(edge_pairs)
                total_edge_sim = (
                    sum(unpack_floats(edge_sims)) / len(edge_sims) if edge_sims else 0.0
                )

                total_sim = (total_node_sim + total_edge_sim) / 2.0

                if total_sim > best_sim:
                    best_sim = total_sim
                    best_node_mapping = {**node_mapping}
                    best_edge_mapping = {
                        y_edge.key: x_edge.key for y_edge, x_edge in edge_pairs
                    }

        return GraphSim(best_sim, best_node_mapping, best_edge_mapping, {}, {})
