import itertools
from dataclasses import dataclass

import numpy as np
from frozendict import frozendict
from scipy.optimize import quadratic_assignment

from ...helpers import get_logger
from ...model.graph import Graph
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass(slots=True)
class qap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Quadratic Assignment Problem (QAP) solver for graph similarity"""

    node_del_cost: float = 1.0
    node_ins_cost: float = 1.0
    edge_del_cost: float = 1.0
    edge_ins_cost: float = 1.0
    illegal_cost: float = 1e9

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
        b = np.full((dim, dim), self.illegal_cost, dtype=float)

        y2idx = {k: i for i, k in enumerate(y.nodes)}
        x2idx = {k: i for i, k in enumerate(x.nodes)}
        idx2y = {i: k for k, i in y2idx.items()}
        idx2x = {i: k for k, i in x2idx.items()}

        # (fast look‑ups for present / absent edges)
        y_edges = {(y2idx[e.source.key], y2idx[e.target.key]) for e in y.edges.values()}
        x_edges = {(x2idx[e.source.key], x2idx[e.target.key]) for e in x.edges.values()}

        # linear part
        # substitution: real-real
        for (y_key, i), (x_key, j) in itertools.product(y2idx.items(), x2idx.items()):
            b[i, j] = (
                1.0 - node_pair_sims[(y_key, x_key)]
                if (y_key, x_key) in node_pair_sims
                else self.illegal_cost
            )

        # deletion: real-dummy
        for y_key, i in y2idx.items():
            dummy_col = m + i
            b[i, dummy_col] = self.node_del_cost
            a[dummy_col, dummy_col] = 1.0  # selector

        # insertion: dummy-real
        for x_key, j in x2idx.items():
            dummy_row = n + j
            b[dummy_row, j] = self.node_ins_cost
            a[dummy_row, dummy_row] = 1.0  # selector

        # quadratic part
        # real edges of y in A
        for e in y.edges.values():
            i, j = y2idx[e.source.key], y2idx[e.target.key]
            a[i, j] = 1.0

        # real edges of x in B
        # not needed for directed graphs
        # for e in x.edges.values():
        #     i, j = x2idx[e.source.key], x2idx[e.target.key]
        #     b[i, j] = 1.0

        # edge substitution
        for y_edge, x_edge in itertools.product(y.edges.values(), x.edges.values()):
            iy, jy = y2idx[y_edge.source.key], y2idx[y_edge.target.key]
            ix, jx = x2idx[x_edge.source.key], x2idx[x_edge.target.key]
            cost = (
                1.0 - edge_pair_sims[(y_edge.key, x_edge.key)]
                if (y_edge.key, x_edge.key) in edge_pair_sims
                else self.illegal_cost
            )

            # four row/col combinations induced by the permutation
            for r, c in ((iy, ix), (iy, jx), (jy, ix), (jy, jx)):
                # keep the *lowest* cost if collisions happen
                b[r, c] = min(b[r, c], cost)

        # edge deletion/insertion
        for iy, jy in itertools.product(range(n), range(n)):
            y_has = (iy, jy) in y_edges

            for ix, jx in itertools.product(range(m), range(m)):
                x_has = (ix, jx) in x_edges

                if y_has and not x_has:  # deletion
                    cost = self.edge_del_cost
                elif not y_has and x_has:  # insertion
                    cost = self.edge_ins_cost
                else:  # no op
                    continue

                for r, c in ((iy, ix), (iy, jx), (jy, ix), (jy, jx)):
                    b[r, c] = min(b[r, c], cost)

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
            (idx2y[row], idx2x[col])
            for row, col in enumerate(res.col_ind)
            if row < n and col < m and (idx2y[row], idx2x[col]) in node_pair_sims
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
