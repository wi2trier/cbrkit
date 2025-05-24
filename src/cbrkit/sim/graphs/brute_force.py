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
        mapped_nodes = dict(zip(y_nodes, x_nodes, strict=True))

        # if one node can't be matched to another, skip this permutation
        for y_key, x_key in mapped_nodes.items():
            if not self.node_matcher(y.nodes[y_key].value, x.nodes[x_key].value):
                return None

        node_pair_sims = self.node_pair_similarities(x, y, list(mapped_nodes.items()))

        # compute edge similarities among matched nodes
        mapped_edges: dict[K, K] = {}

        for y_key, y_edge in y.edges.items():
            # only consider edges whose both endpoints are in our subset
            if y_edge.source.key in mapped_nodes and y_edge.target.key in mapped_nodes:
                y_source = mapped_nodes[y_edge.source.key]
                y_target = mapped_nodes[y_edge.target.key]

                for x_key, x_edge in x.edges.items():
                    if (
                        x_edge.source.key == y_source
                        and x_edge.target.key == y_target
                        and self.edge_matcher(y_edge.value, x_edge.value)
                    ):
                        mapped_edges[y_key] = x_key
                        break

        # batch sim for edges
        edge_pair_sims = self.edge_pair_similarities(
            x,
            y,
            node_pair_sims,
            list(mapped_edges.items()),
        )

        return self.similarity(
            x,
            y,
            frozendict(mapped_nodes),
            frozendict(mapped_edges),
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
                    sim_candidate = self.expand(x, y, y_candidates, x_candidates)

                    if sim_candidate and sim_candidate.value > best_sim.value:
                        best_sim = sim_candidate

        return best_sim
