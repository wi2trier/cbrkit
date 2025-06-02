from dataclasses import dataclass
from typing import Literal

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_array
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from ...helpers import get_logger
from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)

__all__ = ["lsap"]


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
@dataclass(slots=True)
class lsap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    variant: Literal["sparse", "dense"] = "dense"

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        idx2y = {idx: y_key for idx, y_key in enumerate(y.nodes.keys())}
        idx2x = {idx: x_key for idx, x_key in enumerate(x.nodes.keys())}

        try:
            if self.variant == "dense":
                # only needed for linear assignment with positive costs
                # penalty = 2.0 * len(y.nodes) * len(x.nodes)
                cost = np.array(
                    [
                        [
                            -1.0 - node_pair_sims[(y_key, x_key)]
                            if (y_key, x_key) in node_pair_sims
                            else 0.0
                            for x_key in x.nodes.keys()
                        ]
                        for y_key in y.nodes.keys()
                    ]
                )
                row_indexes, col_indexes = linear_sum_assignment(cost)
                node_mapping = frozendict(
                    (idx2y[row_idx], idx2x[col_idx])
                    for row_idx, col_idx in zip(row_indexes, col_indexes, strict=True)
                    if cost[row_idx, col_idx] < 0.0
                )

            elif self.variant == "sparse":
                cost = np.array(
                    [
                        [
                            # From the documentation:
                            # We require that weights are non-zero only to avoid issues with the handling of explicit zeros when converting between different sparse representations.
                            # Zero weights can be handled by adding a constant to all weights, so that the resulting matrix contains no zeros.
                            -1.0 - node_pair_sims[(y_key, x_key)]
                            if (y_key, x_key) in node_pair_sims
                            else 0.0
                            for x_key in x.nodes.keys()
                        ]
                        for y_key in y.nodes.keys()
                    ]
                )
                biadjacency = csr_array(cost)
                row_indexes, col_indexes = min_weight_full_bipartite_matching(
                    biadjacency
                )
                node_mapping = frozendict(
                    (idx2y[row_idx], idx2x[col_idx])
                    for row_idx, col_idx in zip(row_indexes, col_indexes, strict=True)
                )

            else:
                raise ValueError(f"Invalid variant '{self.variant}'.")

        except ValueError as e:
            logger.warning(f"Failed to compute LAP mapping for two graphs: {e}")

            return GraphSim(
                0.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
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
