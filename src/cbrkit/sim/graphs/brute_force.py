import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from frozendict import frozendict

from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim


@dataclass(slots=True)
class brute_force[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """
    Computes the similarity between two graphs by exhaustively computing all possible mappings
    and selecting the one with the highest similarity score.

    Args:
        node_sim_func: A similarity function for graph nodes that receives two node objects.
        edge_sim_func: A similarity function for graph edges that receives two edge objects.

    Returns:
        The similarity between the two graphs and the best mapping.
    """

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        y_nodes: Sequence[K],
        x_nodes: Sequence[K],
    ) -> GraphSim[K] | None:
        node_mapping = frozendict(zip(y_nodes, x_nodes, strict=True))

        # if one node can't be matched to another, skip this permutation
        for y_key, x_key in node_mapping.items():
            if not self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value):
                return None

        edge_mapping = self.induced_edge_mapping(x, y, node_mapping)

        node_pair_sims = self.node_pair_similarities(x, y, list(node_mapping.items()))
        edge_pair_sims = self.edge_pair_similarities(
            x, y, node_pair_sims, list(edge_mapping.items())
        )

        return self.similarity(
            x,
            y,
            frozendict(node_mapping),
            frozendict(edge_mapping),
            frozendict(node_pair_sims),
            frozendict(edge_pair_sims),
        )

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        y_node_keys = list(y.nodes.keys())
        x_node_keys = list(x.nodes.keys())
        best_sim: GraphSim[K] = GraphSim(
            0.0, frozendict(), frozendict(), frozendict(), frozendict()
        )

        # iterate all possible subset sizes of query (up to target size)
        for k in range(1, min(len(y_node_keys), len(x_node_keys)) + 1):
            # all subsets of query nodes of size k
            for y_candidates in itertools.combinations(y_node_keys, k):
                # all injective mappings from this subset to target nodes
                for x_candidates in itertools.permutations(x_node_keys, k):
                    next_sim = self.expand(x, y, y_candidates, x_candidates)

                    if next_sim and (
                        next_sim.value > best_sim.value
                        or (
                            next_sim.value >= best_sim.value
                            and (
                                len(next_sim.node_mapping) > len(best_sim.node_mapping)
                                or (
                                    len(next_sim.edge_mapping)
                                    > len(best_sim.edge_mapping)
                                )
                            )
                        )
                    ):
                        best_sim = next_sim

        return best_sim
