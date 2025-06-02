import itertools
from dataclasses import dataclass

import numpy as np
from frozendict import frozendict
from scipy.optimize import quadratic_assignment

from ...helpers import get_logger
from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)

__all__ = ["qap"]


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass(slots=True)
class qap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Quadratic Assignment Problem (QAP) solver for graph similarity

    Currently not functional, the generated mappings are not correct.
    """

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        n = len(y.nodes)
        m = len(x.nodes)
        dim = n + m
        a = np.zeros((dim, dim), dtype=float)
        b = np.zeros((dim, dim), dtype=float)

        y2idx = {k: i for i, k in enumerate(y.nodes)}
        x2idx = {k: i for i, k in enumerate(x.nodes)}
        idx2y = {i: k for k, i in y2idx.items()}
        idx2x = {i: k for k, i in x2idx.items()}

        # put 1 on every real-node loop of a
        # encode substitution / deletion cost on the corresponding loop of b
        for i in idx2y.keys():
            a[i, i] = 1.0  # selector
            b[m + i, m + i] = 1.0  # deletion

        for j in idx2x.keys():
            b[j, j] = 1.0  # selector
            a[n + j, n + j] = 1.0  # insertion

        # substitution cost (real-real loops)
        for (y_key, i), (x_key, j) in itertools.product(y2idx.items(), x2idx.items()):
            b[i, j] = (
                1 - node_pair_sims[(y_key, x_key)]
                if (y_key, x_key) in node_pair_sims
                else 1e9
            )

        # real edge in y, deletion cost when mapped to two dummies
        for e in y.edges.values():
            i, j = y2idx[e.source.key], y2idx[e.target.key]
            b[m + i, m + j] = 1
            b[m + j, m + i] = 1

        # real edge in x, insertion cost when mapped from two dummies
        for e in x.edges.values():
            i, j = x2idx[e.source.key], x2idx[e.target.key]
            a[n + i, n + j] = 1
            a[n + j, n + i] = 1

        # real-real pairs, substitution cost
        for y_edge, x_edge in itertools.product(y.edges.values(), x.edges.values()):
            i, j = x2idx[x_edge.source.key], x2idx[x_edge.target.key]
            b[i, j] = (
                1 - edge_pair_sims[(y_edge.key, x_edge.key)]
                if (y_edge.key, x_edge.key) in edge_pair_sims
                else 1e9
            )

        try:
            res = quadratic_assignment(a, b, method="faq")
        except ValueError as e:
            logger.warning(f"Failed to compute QAP mapping for two graphs: {e}")

            return GraphSim(
                0.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
            )

        # only consider substitutions of real nodes
        node_mapping = frozendict(
            (idx2y[y_idx], idx2x[x_idx])
            for y_idx, x_idx in enumerate(res.col_ind)
            if y_idx < n
            and x_idx < m
            and (idx2y[y_idx], idx2x[x_idx]) in node_pair_sims
        )

        edge_mapping = self.induced_edge_mapping(x, y, node_mapping)

        return self.similarity(
            x,
            y,
            node_mapping,
            edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )
