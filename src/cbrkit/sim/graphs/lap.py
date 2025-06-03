import itertools
from dataclasses import dataclass

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment

from ...helpers import get_logger
from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim, PairSim

logger = get_logger(__name__)

__all__ = ["lap"]


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass(slots=True)
class lap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    node_del_cost: float = 2.0
    node_ins_cost: float = 0.0
    edge_del_cost: float = 2.0
    edge_ins_cost: float = 0.0
    # 1.0 gives an upper bound, 0.5 gives a lower bound
    # approximation is better with a lower bound
    # since we compute real edit costs at the end anyway,
    # we can use a lower bound
    edge_edit_factor: float = 0.5
    print_matrix: bool = False

    def connected_edges(self, g: Graph[K, N, E, G], n: K) -> set[K]:
        return {
            e.key for e in g.edges.values() if n == e.source.key or n == e.target.key
        }

    def edge_sub_cost(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        x_node: K,
        y_node: K,
        edge_pair_sims: PairSim[K],
    ) -> float:
        y_edges = self.connected_edges(y, y_node)
        x_edges = self.connected_edges(x, x_node)

        rows = len(y_edges)
        cols = len(x_edges)
        dim = rows + cols

        row2y = {r: k for r, k in enumerate(y_edges)}
        col2x = {c: k for c, k in enumerate(x_edges)}

        cost = np.full((dim, dim), np.inf, dtype=float)
        # empty quadrant
        cost[rows:, cols:] = 0.0
        # deletion diagonal
        np.fill_diagonal(cost[rows:, :cols], self.edge_del_cost)
        # insertion diagonal
        np.fill_diagonal(cost[:rows, cols:], self.edge_ins_cost)

        for r, c in itertools.product(range(rows), range(cols)):
            if (
                (y_key := row2y.get(r))
                and (x_key := col2x.get(c))
                and (sim := edge_pair_sims.get((y_key, x_key)))
            ):
                cost[r, c] = 1.0 - sim

        row_idx, col_idx = linear_sum_assignment(cost)

        return cost[row_idx, col_idx].sum()

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        rows = len(y.nodes)
        cols = len(x.nodes)
        dim = rows + cols

        row2y = {r: k for r, k in enumerate(y.nodes.keys())}
        col2x = {c: k for c, k in enumerate(x.nodes.keys())}

        # the cost matrix looks like this:
        # upper left: substitution
        # upper right: deletion
        # lower left: insertion
        # lower right: empty

        # first initialize the matrix with negative scores
        cost = np.full((dim, dim), np.inf, dtype=float)

        # set the empty quadrant to zero
        cost[rows:, cols:] = 0.0

        # deletion diagonal
        for i in range(rows):
            cost[i, cols + i] = self.node_del_cost + (
                self.edge_edit_factor
                * self.edge_del_cost
                * len(self.connected_edges(y, row2y[i]))
            )

        # insertion diagonal
        for i in range(cols):
            cost[rows + i, i] = self.node_ins_cost + (
                self.edge_edit_factor
                * self.edge_ins_cost
                * len(self.connected_edges(x, col2x[i]))
            )

        # substitution quadrant
        for r, c in itertools.product(range(rows), range(cols)):
            if (
                (y_key := row2y.get(r))
                and (x_key := col2x.get(c))
                and (sim := node_pair_sims.get((y_key, x_key)))
            ):
                node_sub_cost = 1.0 - sim
                edge_sub_cost = self.edge_sub_cost(
                    x,
                    y,
                    x_key,
                    y_key,
                    edge_pair_sims,
                )
                cost[r, c] = node_sub_cost + (self.edge_edit_factor * edge_sub_cost)

        if self.print_matrix:
            with np.printoptions(linewidth=10000):
                logger.info(f"Cost matrix:\n{cost}\n")

        row_idx, col_idx = linear_sum_assignment(cost)

        node_mapping = frozendict(
            (row2y[r], col2x[c])
            for r, c in zip(row_idx, col_idx, strict=True)
            # only consider substitutions
            if (
                r < rows
                and c < cols
                and r in row2y
                and c in col2x
                and cost[r, c] < np.inf
            )
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
