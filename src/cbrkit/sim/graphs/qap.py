from dataclasses import dataclass
from typing import Literal

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment, quadratic_assignment

from ...helpers import get_logger
from ...model.graph import Graph
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)


@dataclass(slots=True)
class qap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Quadratic Assignment Problem (QAP) solver for graph similarity.

    This implementation uses QAP to find the optimal graph matching
    considering both node and edge similarities.
    """

    illegal_cost: float = 1e9
    method: Literal["faq", "2opt"] = "faq"
    rng: np.random.Generator | None = None

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        n = len(y.nodes)
        m = len(x.nodes)

        # Special cases
        if n == 0 and m == 0:
            return GraphSim(
                1.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
            )

        if n == 0 or m == 0:
            return GraphSim(
                0.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
            )

        # If no edges, use linear assignment
        if len(y.edges) == 0 and len(x.edges) == 0:
            return self._solve_lap(x, y, node_pair_sims, edge_pair_sims)

        # Create index mappings
        y_nodes = list(y.nodes.keys())
        x_nodes = list(x.nodes.keys())
        y2idx = {k: i for i, k in enumerate(y_nodes)}
        x2idx = {k: i for i, k in enumerate(x_nodes)}

        # QAP formulation
        # We use max(n, m) to handle different sized graphs
        size = max(n, m)

        # Matrix A: adjacency of query graph
        a = np.zeros((size, size), dtype=float)
        for e in y.edges.values():
            i = y2idx[e.source.key]
            j = y2idx[e.target.key]
            if i < size and j < size:
                a[i, j] = 1.0

        # Matrix B: combined adjacency and cost matrix for case graph
        b = np.zeros((size, size), dtype=float)

        # First, encode the case graph structure
        x_edges = {}
        for e in x.edges.values():
            i = x2idx[e.source.key]
            j = x2idx[e.target.key]
            if i < size and j < size:
                x_edges[(i, j)] = e.key
                b[i, j] = 1.0

        # Build query edge mapping
        y_edges = {}
        for e in y.edges.values():
            i = y2idx[e.source.key]
            j = y2idx[e.target.key]
            if i < size and j < size:
                y_edges[(i, j)] = e.key

        # Create the cost matrix C that combines node and edge costs
        # For QAP, we need to encode costs into matrix B
        # The standard approach is to use: B_new = B * edge_weight + diag(node_costs)

        # Node cost matrix
        node_costs = np.full((size, size), self.illegal_cost, dtype=float)

        # Fill in node substitution costs where valid matches exist
        for (y_key, x_key), sim in node_pair_sims.items():
            i = y2idx[y_key]
            j = x2idx[x_key]
            if i < size and j < size:
                node_costs[i, j] = 1.0 - sim

        # Node deletion costs (query to dummy)
        for i in range(n):
            for j in range(m, size):
                node_costs[i, j] = self.node_del_cost

        # Node insertion costs (dummy to case)
        for i in range(n, size):
            for j in range(m):
                node_costs[i, j] = self.node_ins_cost

        # Dummy to dummy - zero cost
        for i in range(n, size):
            for j in range(m, size):
                node_costs[i, j] = 0.0

        # Edge cost matrix
        edge_cost_matrix = np.zeros((size, size), dtype=float)

        # For each edge position in the matrices
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue

                # Default costs based on what exists
                if (i, j) in x_edges:
                    # Edge exists in x - default is deletion cost
                    edge_cost_matrix[i, j] = self.edge_del_cost

                    # Check for substitution
                    x_edge = x_edges[(i, j)]
                    for (yi, yj), y_edge in y_edges.items():
                        if (y_edge, x_edge) in edge_pair_sims:
                            cost = 1.0 - edge_pair_sims[(y_edge, x_edge)]
                            edge_cost_matrix[i, j] = min(edge_cost_matrix[i, j], cost)
                else:
                    # No edge in x
                    if a.sum() > 0:  # Query has edges
                        edge_cost_matrix[i, j] = self.edge_ins_cost
                    else:
                        edge_cost_matrix[i, j] = 0.0

        # For QAP, we need to solve min trace(X^T A X B) + trace(X^T C)
        # where C is the node cost matrix
        # Standard QAP solvers don't directly support the linear term
        # So we use the augmented formulation with node costs on diagonal

        # Create augmented adjacency matrix with self-loops
        a_aug = a + np.eye(size)

        # Create cost matrix B that combines edge costs and node costs
        # The diagonal of B will contain node costs
        b_cost = np.zeros((size, size), dtype=float)

        # Copy node costs to B's structure
        # This encodes the linear assignment costs into the QAP formulation
        for i in range(size):
            for j in range(size):
                if i == j:
                    # Diagonal: node assignment cost
                    b_cost[i, j] = node_costs[i, j]
                else:
                    # Off-diagonal: edge costs (simplified for now)
                    b_cost[i, j] = edge_cost_matrix[i, j]

        try:
            # Solve QAP
            res = quadratic_assignment(
                a_aug, b_cost, method=self.method, options={"rng": self.rng}
            )

            # Extract node mapping
            node_mapping = {}
            for i in range(n):
                j = res.col_ind[i]
                if j < m:  # Mapped to real case node
                    y_key = y_nodes[i]
                    x_key = x_nodes[j]
                    if (y_key, x_key) in node_pair_sims:
                        node_mapping[y_key] = x_key

            # If QAP gives very high cost, it might be forcing invalid mappings
            # In such cases, fall back to LAP
            if res.fun > self.illegal_cost / 2:
                return self._solve_lap(x, y, node_pair_sims, edge_pair_sims)

            node_mapping = frozendict(node_mapping)

        except ValueError as e:
            logger.warning(f"QAP solver failed: {e}")
            node_mapping = frozendict()

        # Compute induced edge mapping
        edge_mapping = self.induced_edge_mapping(x, y, node_mapping)

        # Compute final similarity
        return self.similarity(
            x,
            y,
            node_mapping,
            edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )

    def _solve_lap(self, x, y, node_pair_sims, edge_pair_sims):
        """Solve using Linear Assignment Problem for partial mappings."""
        n = len(y.nodes)
        m = len(x.nodes)

        # For partial mappings, we need to handle the case where n != m
        # We'll create a cost matrix that allows deletion/insertion
        max_size = max(n, m)

        # Build extended cost matrix
        c = np.full((max_size, max_size), self.node_del_cost, dtype=float)

        # Fill actual node similarities
        y_keys = list(y.nodes.keys())
        x_keys = list(x.nodes.keys())

        for (y_key, x_key), sim in node_pair_sims.items():
            i = y_keys.index(y_key)
            j = x_keys.index(x_key)
            c[i, j] = 1.0 - sim

        # For query nodes that can't match any case node, all costs are deletion cost
        # For positions beyond actual nodes, cost is already set to deletion cost

        # Solve LAP
        row_ind, col_ind = linear_sum_assignment(c)

        # Extract mapping
        node_mapping = {}
        for idx, i in enumerate(row_ind):
            if i < n and col_ind[idx] < m:
                y_key = y_keys[i]
                x_key = x_keys[col_ind[idx]]
                # Only include if it's a valid match
                if (y_key, x_key) in node_pair_sims:
                    node_mapping[y_key] = x_key

        # Get induced edge mapping
        edge_mapping = self.induced_edge_mapping(x, y, frozendict(node_mapping))

        return self.similarity(
            x, y, frozendict(node_mapping), edge_mapping, node_pair_sims, edge_pair_sims
        )
