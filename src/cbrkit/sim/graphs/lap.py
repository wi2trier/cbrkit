import itertools
import logging
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
import scipy

from ...helpers import frozendict
from ...model.graph import Graph
from ...model.similarity import GraphSim
from ...schema import serializable
from ...typing import NumpyArray
from ._base import BaseGraphEditFunc, BaseGraphSimFunc, PairSim, PairSimilarities

__all__ = ["LAP", "BaseLAP", "LAPEdgeHandling", "incident_edges"]

logger = logging.getLogger(__name__)

type LAPEdgeHandling = Literal["ignore", "worstcase", "greedy", "optimal"]


def incident_edges[K: Hashable, N, E, G](g: Graph[K, N, E, G]) -> dict[K, list[K]]:
    """Maps every node to the keys of its incident edges in a single pass.

    The order follows the graph, so that the greedy tie-breaking downstream does
    not depend on set iteration order.
    A self loop is listed once, and the endpoints of an edge are included even if a
    subgraph lacks them.
    """
    adjacency: dict[K, list[K]] = {key: [] for key in g.nodes}

    for edge in g.edges.values():
        adjacency.setdefault(edge.source.key, []).append(edge.key)

        if edge.target.key != edge.source.key:
            adjacency.setdefault(edge.target.key, []).append(edge.key)

    return adjacency


# https://jack.valmadre.net/notes/2020/12/08/non-perfect-linear-assignment/
# not slotted, so it can be mixed into other slotted dataclasses
@dataclass(frozen=True, kw_only=True)
class BaseLAP[K: Hashable, N, E, G](BaseGraphEditFunc[K, N, E, G]):
    """Base class for building LAP cost matrices with configurable edge handling.

    Args:
        edge_handling: How the edges of a node pair add to its cost,
            greedily, by solving an inner assignment, as the worst case, or not at all.
        edge_edit_factor: Factor of the edge costs of a node pair,
            1.0 gives an upper bound and 0.5 a lower bound, which approximates better.
    """

    edge_handling: LAPEdgeHandling = "greedy"
    edge_edit_factor: float = 0.5

    def edge_sub_cost_greedy(
        self,
        x_edges: Sequence[K],
        y_edges: Sequence[K],
        edge_pair_sims: PairSim[K],
    ) -> float:
        """BranchFast algorithm without solving an inner LAP problem.

        - Substitutions are taken greedily in descending similarity order
        - Unmatched y‑edges are deletions, unmatched x‑edges are insertions.
        """

        # trivial fast‑path
        if not y_edges and not x_edges:
            return 0.0

        # candidate substitutions: (cost, y_key, x_key)
        candidates: list[tuple[float, K, K]] = [
            (self.edge_weight * (1.0 - edge_pair_sims[(y_key, x_key)]), y_key, x_key)
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
        cost += (len(y_edges) - len(matched_y)) * self.edge_deletion_cost
        cost += (len(x_edges) - len(matched_x)) * self.edge_insertion_cost

        return cost

    def edge_sub_cost_optimal(
        self,
        x_edges: Sequence[K],
        y_edges: Sequence[K],
        edge_pair_sims: PairSim[K],
    ) -> float:
        """Branch algorithm solving an inner LAP problem."""

        if not y_edges and not x_edges:
            return 0.0

        rows = len(y_edges)
        cols = len(x_edges)
        dim = rows + cols

        row2y = dict(enumerate(y_edges))
        col2x = dict(enumerate(x_edges))

        cost_matrix = np.full((dim, dim), np.inf, dtype=float)
        # empty quadrant
        cost_matrix[rows:, cols:] = 0.0
        # deletion diagonal
        np.fill_diagonal(cost_matrix[:rows, cols:], self.edge_deletion_cost)
        # insertion diagonal
        np.fill_diagonal(cost_matrix[rows:, :cols], self.edge_insertion_cost)

        for r, c in itertools.product(range(rows), range(cols)):
            # Access by index to avoid falsy-key short-circuiting (e.g., 0 or "").
            y_key = row2y[r]
            x_key = col2x[c]
            sim = edge_pair_sims.get((y_key, x_key))
            if sim is not None:
                cost_matrix[r, c] = self.edge_weight * (1.0 - sim)

        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

        return cost_matrix[row_idx, col_idx].sum()

    def build_cost_matrix(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_pair_sims: PairSim[K],
        edge_pair_sims: PairSim[K],
    ) -> NumpyArray:
        """Builds a normalized cost matrix for the linear assignment problem."""
        rows = len(query.nodes)
        cols = len(case.nodes)
        dim = rows + cols

        upper_bound = self.cost_upper_bound(case, query)

        if upper_bound == 0.0:
            return np.zeros((dim, dim), dtype=float)

        normalization_factor = 1.0 / upper_bound

        row2y = dict(enumerate(query.nodes))
        col2x = dict(enumerate(case.nodes))
        y_adjacency = incident_edges(query)
        x_adjacency = incident_edges(case)

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
            if self.edge_handling == "ignore":
                edge_deletion_cost = 0.0
            else:
                edge_deletion_cost = self.edge_deletion_cost * len(
                    y_adjacency[row2y[i]]
                )

            cost_matrix[i, cols + i] = (
                self.node_deletion_cost + (self.edge_edit_factor * edge_deletion_cost)
            ) * normalization_factor

        # insertion diagonal
        for i in range(cols):
            if self.edge_handling == "ignore":
                edge_insertion_cost = 0.0
            else:
                edge_insertion_cost = self.edge_insertion_cost * len(
                    x_adjacency[col2x[i]]
                )

            cost_matrix[rows + i, i] = (
                self.node_insertion_cost + (self.edge_edit_factor * edge_insertion_cost)
            ) * normalization_factor

        # substitution quadrant
        for r, c in itertools.product(range(rows), range(cols)):
            y_key = row2y[r]
            x_key = col2x[c]
            sim = node_pair_sims.get((y_key, x_key))

            if sim is None:
                continue

            node_sub_cost = self.node_weight * (1.0 - sim)
            x_edges = x_adjacency[x_key]
            y_edges = y_adjacency[y_key]

            if self.edge_handling == "greedy":
                edge_sub_cost = self.edge_sub_cost_greedy(
                    x_edges, y_edges, edge_pair_sims
                )
            elif self.edge_handling == "optimal":
                edge_sub_cost = self.edge_sub_cost_optimal(
                    x_edges, y_edges, edge_pair_sims
                )
            elif self.edge_handling == "worstcase":
                edge_sub_cost = self.edge_deletion_cost * len(
                    y_edges
                ) + self.edge_insertion_cost * len(x_edges)
            else:
                edge_sub_cost = 0.0

            cost_matrix[r, c] = (
                node_sub_cost + (self.edge_edit_factor * edge_sub_cost)
            ) * normalization_factor

        if logger.isEnabledFor(logging.DEBUG):
            with np.printoptions(linewidth=10000):
                logger.debug("Cost matrix:\n%s", cost_matrix)

        return cost_matrix

    def lap_node_mapping(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_pair_sims: PairSim[K],
        edge_pair_sims: PairSim[K],
    ) -> frozendict[K, K]:
        """Solves the assignment problem and keeps its substitutions."""
        cost_matrix = self.build_cost_matrix(
            case, query, node_pair_sims, edge_pair_sims
        )
        row2y = dict(enumerate(query.nodes.keys()))
        col2x = dict(enumerate(case.nodes.keys()))
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost_matrix)

        return frozendict(
            (row2y[int(r)], col2x[int(c)])
            for r, c in zip(row_idx, col_idx, strict=True)
            if cost_matrix[r, c] < np.inf and r in row2y and c in col2x
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class LAP[K: Hashable, N, E, G](BaseLAP[K, N, E, G], BaseGraphSimFunc[K, N, E, G]):
    """Graph similarity using the Linear Assignment Problem (LAP).

    Examples:
        >>> from ...model.graph import from_dict
        >>> graph = from_dict({
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2"}},
        ... })
        >>> sim = LAP(node_measure=lambda x, y: 1.0 if x == y else 0.0)
        >>> [result.value for result in sim([(graph, graph)])]
        [1.0]
    """

    kind: Literal["lap"] = field(default="lap", repr=False)

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        node_mapping = self.lap_node_mapping(case, query, sims.nodes, sims.edges)

        return self.similarity(case, query, node_mapping, sims)
