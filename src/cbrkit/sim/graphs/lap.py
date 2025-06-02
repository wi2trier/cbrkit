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
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)

__all__ = ["lap"]


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass(slots=True)
class lap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        # TODO: Edge constraints not respected here!
        # maybe fall back to mapping nodes only and using the induced edge mapping?
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        ny, ey = len(y.nodes), len(y.edges)
        nx, ex = len(x.nodes), len(x.edges)
        rows = ny + ey
        cols = nx + ex
        dim = rows + cols

        row2y_nodes = {r: k for r, k in enumerate(y.nodes.keys())}
        row2y_edges = {ny + r: k for r, k in enumerate(y.edges.keys())}

        col2x_nodes = {c: k for c, k in enumerate(x.nodes.keys())}
        col2x_edges = {nx + c: k for c, k in enumerate(x.edges.keys())}

        # the cost matrix looks like this:
        # upper left: substitution
        # upper right: deletion
        # lower left: insertion
        # lower right: empty
        # each quadrant contains cost for nodes and edges with nodes coming first

        # first initialize the matrix with negative scores
        cost = np.full((dim, dim), np.inf, dtype=float)

        # then set the empty quadrant to zero
        cost[rows:, cols:] = 0.0

        # then set the diagonals of deletion and insertion quadrants to zero
        np.fill_diagonal(cost[rows:, :cols], 1.0)
        np.fill_diagonal(cost[:rows, cols:], 1.0)

        # then fill the substitution quadrant with node and edge similarities
        for r, c in itertools.product(range(rows), range(cols)):
            if (
                (y_key := row2y_nodes.get(r))
                and (x_key := col2x_nodes.get(c))
                and (sim := node_pair_sims.get((y_key, x_key)))
            ):
                cost[r, c] = 1.0 - sim

            elif (
                (y_key := row2y_edges.get(r))
                and (x_key := col2x_edges.get(c))
                and (sim := edge_pair_sims.get((y_key, x_key)))
            ):
                cost[r, c] = 1.0 - sim

        # for debugging purposes, you can uncomment the following lines
        # with np.printoptions(threshold=np.inf, linewidth=1e12):
        #     print(f"Cost matrix:\n{cost}")

        try:
            row_idx, col_idx = linear_sum_assignment(cost)
            node_mapping: dict[K, K] = {}
            edge_mapping: dict[K, K] = {}

            for r, c in zip(row_idx, col_idx, strict=True):
                # only consider substitutions
                if r < rows and c < cols:
                    # nodes
                    if r in row2y_nodes and c in col2x_nodes and cost[r, c] < np.inf:
                        node_mapping[row2y_nodes[r]] = col2x_nodes[c]

                    # edges
                    elif r in row2y_edges and c in col2x_edges and cost[r, c] < np.inf:
                        edge_mapping[row2y_edges[r]] = col2x_edges[c]

        except ValueError as e:
            logger.warning(f"Failed to compute LAP mapping for two graphs: {e}")

            return GraphSim(
                0.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
            )

        return self.similarity(
            x,
            y,
            frozendict(node_mapping),
            frozendict(edge_mapping),
            node_pair_sims,
            edge_pair_sims,
        )
