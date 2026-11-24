import logging
import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

import numpy as np
import scipy
from pydantic import Field

from ...helpers import frozendict
from ...model.graph import Graph
from ...model.similarity import GraphSim
from ...schema import serializable
from ...typing import NumpyArray
from ._base import BaseGraphSimFunc, PairSimilarities, extend_maximal
from .lap import BaseLAP

logger = logging.getLogger(__name__)

# The largest node limit of HiGHS, which is its default.
UNLIMITED_NODES = int(np.iinfo(np.int32).max)

__all__ = ["QAP"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class QAP[K: Hashable, N, E, G](BaseLAP[K, N, E, G], BaseGraphSimFunc[K, N, E, G]):
    """Graph similarity as a quadratic assignment problem.

    Mapping two graphs onto each other is a quadratic assignment problem, since the
    contribution of an edge depends on where both of its endpoints are mapped.
    `LAP` sidesteps the quadratic term by folding an estimate of the edge cost into
    the node costs, which makes it fast but approximate.
    This function keeps the quadratic term and states it as a binary linear program,
    which is the standard linearization of the problem and is solved exactly.

    One variable is created per legal node pair and one per legal edge pair.
    The constraints are exactly the legality conditions of
    [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005):
    every node and edge of either graph takes part in at most one pair, which makes
    the mapping partial and injective, and an edge pair may only be selected if both
    of its endpoint pairs are, which makes the edges induced by the nodes.

    The result is therefore identical to `Astar` and `BruteForce`, while the branch
    and bound of the solver is usually much faster than searching the mappings.

    Whenever a budget stops the solver before it proves optimality, the mapping of
    `LAP` is computed as well and returned if it is better, and the result is not
    flagged as `exact`.

    Args:
        max_expansions: Number of branch and bound nodes after which the solver returns
            the best mapping found so far.
            Unlike `timeout`, this budget gives the same result on every run.
            Unbounded by default.
        timeout: Seconds after which the solver returns the best mapping found so
            far, which then depends on the load of the machine.
            Unbounded by default.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_measure = lambda n1, n2: 1.0 if n1 == n2 else 0.0
        >>> data_x = {
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> data_y = {
        ...     "nodes": {"1": "A", "2": "C"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> graph_x = from_dict(data_x)
        >>> graph_y = from_dict(data_y)
        >>> sim = QAP(node_measure=node_measure)
        >>> sim([(graph_x, graph_x)])[0]
        GraphSim(value=1.0,
            similarities=frozendict({'1': 1.0, '2': 1.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 1.0}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=True)
        >>> sim([(graph_x, graph_y)])[0]
        GraphSim(value=0.5,
            similarities=frozendict({'1': 1.0, '2': 0.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 0.5}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=True)
    """

    max_expansions: Annotated[int, Field(ge=0)] | None = None
    timeout: Annotated[float, Field(ge=0)] | None = None
    kind: Literal["qap"] = field(default="qap", repr=False)

    def assignment_constraint(
        self,
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
    ) -> scipy.optimize.LinearConstraint:
        """Restricts every graph element to at most one of the pairs it occurs in.

        This is what makes the mapping partial and injective, for the nodes as well
        as for the edges.
        """
        rows: dict[tuple[str, K], int] = {}
        row_indices: list[int] = []
        column_indices: list[int] = []

        def row(group: str, key: K) -> int:
            return rows.setdefault((group, key), len(rows))

        for column, (y_key, x_key) in enumerate(node_pairs):
            row_indices += [row("query node", y_key), row("case node", x_key)]
            column_indices += [column, column]

        for offset, (y_key, x_key) in enumerate(edge_pairs):
            column = len(node_pairs) + offset
            row_indices += [row("query edge", y_key), row("case edge", x_key)]
            column_indices += [column, column]

        matrix = scipy.sparse.coo_array(
            (np.ones(len(row_indices)), (row_indices, column_indices)),
            shape=(len(rows), len(node_pairs) + len(edge_pairs)),
        )

        return scipy.optimize.LinearConstraint(matrix, 0, 1)

    def endpoint_constraint(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
    ) -> scipy.optimize.LinearConstraint:
        """Allows an edge pair only if both of its endpoint pairs are selected.

        This is the quadratic part of the problem, expressed as `edge - endpoint <= 0`
        once for the source and once for the target of the edge.
        """
        node_columns = {pair: column for column, pair in enumerate(node_pairs)}
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []

        for offset, (y_key, x_key) in enumerate(edge_pairs):
            y_edge = query.edges[y_key]
            x_edge = case.edges[x_key]
            endpoints = (
                (y_edge.source.key, x_edge.source.key),
                (y_edge.target.key, x_edge.target.key),
            )

            for position, endpoint in enumerate(endpoints):
                row_indices += [2 * offset + position] * 2
                column_indices += [len(node_pairs) + offset, node_columns[endpoint]]
                values += [1.0, -1.0]

        matrix = scipy.sparse.coo_array(
            (values, (row_indices, column_indices)),
            shape=(2 * len(edge_pairs), len(node_pairs) + len(edge_pairs)),
        )

        return scipy.optimize.LinearConstraint(matrix, -np.inf, 0)

    def optimize(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
        objective: NumpyArray,
    ) -> tuple[NumpyArray | None, bool]:
        """Selects the pairs that maximize the total similarity.

        Returns the selection, if any was found, and whether it is proven optimal.
        """
        constraints = [self.assignment_constraint(node_pairs, edge_pairs)]

        if edge_pairs:
            constraints.append(
                self.endpoint_constraint(case, query, node_pairs, edge_pairs)
            )

        result = scipy.optimize.milp(
            # The solver minimizes, so the similarities are negated to maximize them.
            -objective,
            integrality=np.ones(len(objective), dtype=int),
            bounds=scipy.optimize.Bounds(0, 1),
            constraints=constraints,
            options={
                "node_limit": UNLIMITED_NODES
                if self.max_expansions is None
                else self.max_expansions,
                "time_limit": math.inf if self.timeout is None else self.timeout,
            },
        )

        if result.x is None:
            logger.warning("No mapping found: %s", result.message)

            return None, False

        if result.status != 0:
            logger.warning("Mapping may be suboptimal: %s", result.message)

        return np.round(result.x).astype(bool), result.status == 0

    def node_mapping(
        self,
        node_pairs: list[tuple[K, K]],
        selected: NumpyArray,
    ) -> frozendict[K, K]:
        """Reads the node mapping off the solution and makes it maximal.

        A pair the solver left out while both of its nodes stayed free is worth
        nothing, because taking it could otherwise only have improved the objective.
        """
        return frozendict(
            extend_maximal(
                {
                    y_key: x_key
                    for (y_key, x_key), active in zip(node_pairs, selected, strict=True)
                    if active
                },
                node_pairs,
            )
        )

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = sims.nodes, sims.edges
        node_pairs = list(node_pair_sims)
        edge_pairs = list(edge_pair_sims)
        # Without node pairs, the empty mapping is the only and thus optimal one.
        selected, optimal = (
            self.optimize(
                case,
                query,
                node_pairs,
                edge_pairs,
                np.array(
                    [self.node_weight * sim for sim in node_pair_sims.values()]
                    + [self.edge_weight * sim for sim in edge_pair_sims.values()]
                ),
            )
            if node_pairs
            else (None, True)
        )
        node_mapping = (
            frozendict[K, K]()
            if selected is None
            else self.node_mapping(node_pairs, selected[: len(node_pairs)])
        )

        node_mappings = [node_mapping]

        if not optimal:
            node_mappings.append(
                self.lap_node_mapping(case, query, node_pair_sims, edge_pair_sims)
            )

        # The edge variables drive the objective, but the mapping is derived from the
        # nodes so that it is maximal for the same reason as above.
        # On a tie, the solution of the solver is kept.
        return max(
            (
                self.similarity(case, query, mapping, sims, exact=optimal)
                for mapping in node_mappings
            ),
            key=lambda sim: sim.value,
        )
