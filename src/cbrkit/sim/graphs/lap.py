import itertools
from dataclasses import dataclass

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment

from ...helpers import get_logger
from ...model.graph import Graph
from ...typing import NumpyArray, SimFunc
from .common import BaseGraphEditFunc, BaseGraphSimFunc, GraphSim, PairSim

logger = get_logger(__name__)


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass
class lap_base[K, N, E, G](BaseGraphEditFunc[K, N, E, G]):
    greedy: bool = True
    # 1.0 gives an upper bound, 0.5 gives a lower bound
    # approximation is better with a lower bound
    edge_edit_factor: float = 0.5
    print_matrix: bool = False

    def connected_edges(self, g: Graph[K, N, E, G], n: K) -> set[K]:
        return {
            e.key for e in g.edges.values() if n == e.source.key or n == e.target.key
        }

    def edge_sub_cost_greedy(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        x_node: K,
        y_node: K,
        edge_pair_sims: PairSim[K],
    ) -> float:
        """BranchFast algorithm without solving an inner LAP problem.

        - Substitutions are taken greedily in descending similarity order
        - Unmatched y‑edges are deletions, unmatched x‑edges are insertions.
        """

        y_edges = list(self.connected_edges(y, y_node))
        x_edges = list(self.connected_edges(x, x_node))

        # trivial fast‑path
        if not y_edges and not x_edges:
            return 0.0

        # candidate substitutions: (cost, y_key, x_key)
        candidates: list[tuple[float, K, K]] = [
            (1.0 - edge_pair_sims[(y_key, x_key)], y_key, x_key)
            for y_key, x_key in itertools.product(y_edges, x_edges)
            if (y_key, x_key) in edge_pair_sims
        ]
        # sort by cheapest cost  ==> highest similarity first
        candidates.sort(key=lambda t: t[0])

        matched_y: set[K] = set()
        matched_x: set[K] = set()
        cost = 0.0

        for c, y_key, x_key in candidates:
            if y_key not in matched_y and x_key not in matched_x:
                matched_y.add(y_key)
                matched_x.add(x_key)
                cost += c  # substitution cost

        # remaining deletions / insertions
        cost += (len(y_edges) - len(matched_y)) * self.edge_del_cost
        cost += (len(x_edges) - len(matched_x)) * self.edge_ins_cost

        return cost

    def edge_sub_cost_optimal(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        x_node: K,
        y_node: K,
        edge_pair_sims: PairSim[K],
    ) -> float:
        """Branch algorithm solving an inner LAP problem."""

        y_edges = self.connected_edges(y, y_node)
        x_edges = self.connected_edges(x, x_node)

        if not y_edges and not x_edges:
            return 0.0

        rows = len(y_edges)
        cols = len(x_edges)
        dim = rows + cols

        row2y = {r: k for r, k in enumerate(y_edges)}
        col2x = {c: k for c, k in enumerate(x_edges)}

        cost_matrix = np.full((dim, dim), np.inf, dtype=float)
        # empty quadrant
        cost_matrix[rows:, cols:] = 0.0
        # deletion diagonal
        np.fill_diagonal(cost_matrix[:rows, cols:], self.edge_del_cost)
        # insertion diagonal
        np.fill_diagonal(cost_matrix[rows:, :cols], self.edge_ins_cost)

        for r, c in itertools.product(range(rows), range(cols)):
            if (
                (y_key := row2y.get(r))
                and (x_key := col2x.get(c))
                and (sim := edge_pair_sims.get((y_key, x_key)))
            ):
                cost_matrix[r, c] = 1.0 - sim

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        return cost_matrix[row_idx, col_idx].sum()

    def build_cost_matrix(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_pair_sims: PairSim[K],
        edge_pair_sims: PairSim[K],
    ) -> NumpyArray:
        rows = len(y.nodes)
        cols = len(x.nodes)
        dim = rows + cols

        cost_upper_bound = (
            len(y.nodes) * self.node_del_cost
            + len(x.nodes) * self.node_ins_cost
            + len(y.edges) * self.edge_del_cost
            + len(x.edges) * self.edge_ins_cost
        )

        if cost_upper_bound == 0.0:
            return np.zeros((dim, dim), dtype=float)

        normalization_factor = 1.0 / cost_upper_bound

        row2y = {r: k for r, k in enumerate(y.nodes.keys())}
        col2x = {c: k for c, k in enumerate(x.nodes.keys())}

        # the cost matrix looks like this:
        # upper left: substitution
        # upper right: deletion
        # lower left: insertion
        # lower right: empty

        # first initialize the matrix with negative scores
        cost_matrix = np.full((dim, dim), np.inf, dtype=float)

        # empty quadrant
        cost_matrix[rows:, cols:] = 0.0

        # deletion diagonal
        for i in range(rows):
            cost_matrix[i, cols + i] = (
                self.node_del_cost
                + (
                    self.edge_edit_factor
                    * self.edge_del_cost
                    * len(self.connected_edges(y, row2y[i]))
                )
            ) * normalization_factor

        # insertion diagonal
        for i in range(cols):
            cost_matrix[rows + i, i] = (
                self.node_ins_cost
                + (
                    self.edge_edit_factor
                    * self.edge_ins_cost
                    * len(self.connected_edges(x, col2x[i]))
                )
            ) * normalization_factor

        # substitution quadrant
        for r, c in itertools.product(range(rows), range(cols)):
            if (
                (y_key := row2y.get(r))
                and (x_key := col2x.get(c))
                and (sim := node_pair_sims.get((y_key, x_key)))
            ):
                node_sub_cost = 1.0 - sim

                if self.greedy:
                    edge_sub_cost = self.edge_sub_cost_greedy(
                        x, y, x_key, y_key, edge_pair_sims
                    )
                else:
                    edge_sub_cost = self.edge_sub_cost_optimal(
                        x, y, x_key, y_key, edge_pair_sims
                    )

                cost_matrix[r, c] = (
                    node_sub_cost + (self.edge_edit_factor * edge_sub_cost)
                ) * normalization_factor

        if self.print_matrix:
            with np.printoptions(linewidth=10000):
                logger.info(f"Cost matrix:\n{cost_matrix}\n")

        return cost_matrix


@dataclass(slots=True)
class lap[K, N, E, G](
    lap_base[K, N, E, G],
    BaseGraphSimFunc[K, N, E, G],
    SimFunc[Graph[K, N, E, G], GraphSim[K]],
):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)
        cost_matrix = self.build_cost_matrix(x, y, node_pair_sims, edge_pair_sims)
        row2y = {r: k for r, k in enumerate(y.nodes.keys())}
        col2x = {c: k for c, k in enumerate(x.nodes.keys())}

        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        node_mapping = frozendict(
            (row2y[r], col2x[c])
            for r, c in zip(row_idx, col_idx, strict=True)
            # only consider substitutions
            if cost_matrix[r, c] < np.inf and r in row2y and c in col2x
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
