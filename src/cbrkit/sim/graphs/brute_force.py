import itertools
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal, override

from ...helpers import frozendict
from ...model.graph import (
    Graph,
)
from ...model.similarity import GraphSim
from ...schema import serializable
from ._base import BaseGraphSimFunc, PairSimilarities

__all__ = ["BruteForce"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class BruteForce[K: Hashable, N, E, G](BaseGraphSimFunc[K, N, E, G]):
    """Computes the similarity between two graphs by exhaustively computing all possible mappings
    and selecting the one with the highest similarity score."""

    kind: Literal["brute_force"] = field(default="brute_force", repr=False)

    def expand(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        y_nodes: Sequence[K],
        x_nodes: Sequence[K],
        sims: PairSimilarities[K],
    ) -> GraphSim[K] | None:
        """Evaluates a single node mapping candidate and returns its similarity."""
        node_mapping = frozendict(zip(y_nodes, x_nodes, strict=True))

        # if one node can't be matched to another, skip this permutation
        for y_key, x_key in node_mapping.items():
            if (y_key, x_key) not in sims.nodes:
                return None

        return self.similarity(case, query, node_mapping, sims, exact=True)

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        y_node_keys = list(query.nodes.keys())
        x_node_keys = list(case.nodes.keys())
        # The empty mapping is a legal candidate and the only one for an empty query.
        best_sim = self.similarity(case, query, frozendict(), sims, exact=True)

        # iterate all possible subset sizes of query (up to target size)
        for k in range(1, min(len(y_node_keys), len(x_node_keys)) + 1):
            # all subsets of query nodes of size k
            for y_candidates in itertools.combinations(y_node_keys, k):
                # all injective mappings from this subset to target nodes
                for x_candidates in itertools.permutations(x_node_keys, k):
                    next_sim = self.expand(
                        case,
                        query,
                        y_candidates,
                        x_candidates,
                        sims,
                    )

                    if next_sim is not None and (
                        next_sim.value > best_sim.value
                        or (
                            next_sim.value >= best_sim.value
                            and (
                                len(next_sim.mapping) > len(best_sim.mapping)
                                or (
                                    len(next_sim.edge_mapping)
                                    > len(best_sim.edge_mapping)
                                )
                            )
                        )
                    ):
                        best_sim = next_sim

        return best_sim
