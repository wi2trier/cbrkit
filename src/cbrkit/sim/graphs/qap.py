from dataclasses import dataclass

import numpy as np
from frozendict import frozendict
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array

from ...helpers import get_logger
from ...model.graph import Graph
from ...typing import NumpyArray, SimFunc
from .common import BaseGraphSimFunc, GraphSim, extend_maximal

logger = get_logger(__name__)

__all__ = ["qap"]


@dataclass(slots=True)
class qap[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Graph similarity as a quadratic assignment problem.

    Mapping two graphs onto each other is a quadratic assignment problem, since the
    contribution of an edge depends on where *both* of its endpoints are mapped.
    `lap` sidesteps the quadratic term by folding an estimate of the edge cost into
    the node costs, which makes it fast but approximate.
    This function keeps the quadratic term and states it as a binary linear program,
    which is the standard linearization of the problem and is solved exactly.

    One variable is created per legal node pair and one per legal edge pair.
    The constraints are exactly the legality conditions of
    [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005):
    every node and edge of either graph takes part in at most one pair, which makes
    the mapping partial and injective, and an edge pair may only be selected if both
    of its endpoint pairs are, which makes the edges induced by the nodes.

    The result is therefore identical to `astar` and `brute_force`, while the branch
    and bound of the solver is usually much faster than searching the mappings.

    Args:
        node_sim_func: A similarity function for node values.
        edge_sim_func: A similarity function for edges.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        time_limit: Seconds after which the solver returns the best mapping found so
            far, which is then no longer guaranteed to be optimal.
            Disabled by default.

    Returns:
        The similarity between a query and a case graph along with the mapping.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
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
        >>> sim = qap(node_sim_func)
        >>> sim(graph_x, graph_x)
        GraphSim(value=1.0,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 1.0}),
            edge_similarities=frozendict.frozendict({'e1': 1.0}))
        >>> sim(graph_x, graph_y)
        GraphSim(value=0.5,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 0.0}),
            edge_similarities=frozendict.frozendict({'e1': 0.5}))
    """

    time_limit: float | None = None

    def assignment_constraint(
        self,
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
    ) -> LinearConstraint:
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

        matrix = coo_array(
            (np.ones(len(row_indices)), (row_indices, column_indices)),
            shape=(len(rows), len(node_pairs) + len(edge_pairs)),
        )

        return LinearConstraint(matrix, 0, 1)

    def endpoint_constraint(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
    ) -> LinearConstraint:
        """Allows an edge pair only if both of its endpoint pairs are selected.

        This is the quadratic part of the problem, expressed as `edge - endpoint <= 0`
        once for the source and once for the target of the edge.
        """
        node_columns = {pair: column for column, pair in enumerate(node_pairs)}
        row_indices: list[int] = []
        column_indices: list[int] = []
        values: list[float] = []

        for offset, (y_key, x_key) in enumerate(edge_pairs):
            y_edge = y.edges[y_key]
            x_edge = x.edges[x_key]
            endpoints = (
                (y_edge.source.key, x_edge.source.key),
                (y_edge.target.key, x_edge.target.key),
            )

            for position, endpoint in enumerate(endpoints):
                row_indices += [2 * offset + position] * 2
                column_indices += [len(node_pairs) + offset, node_columns[endpoint]]
                values += [1.0, -1.0]

        matrix = coo_array(
            (values, (row_indices, column_indices)),
            shape=(2 * len(edge_pairs), len(node_pairs) + len(edge_pairs)),
        )

        return LinearConstraint(matrix, -np.inf, 0)

    def solve(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        node_pairs: list[tuple[K, K]],
        edge_pairs: list[tuple[K, K]],
        objective: NumpyArray,
    ) -> NumpyArray | None:
        """Selects the pairs that maximize the total similarity."""
        constraints = [self.assignment_constraint(node_pairs, edge_pairs)]

        if edge_pairs:
            constraints.append(self.endpoint_constraint(x, y, node_pairs, edge_pairs))

        result = milp(
            # The solver minimizes, so the similarities are negated to maximize them.
            -objective,
            integrality=np.ones(len(objective), dtype=int),
            bounds=Bounds(0, 1),
            constraints=constraints,
            options={"time_limit": self.time_limit} if self.time_limit else None,
        )

        if result.x is None:
            logger.warning(f"No mapping found: {result.message}")

            return None

        if result.status != 0:
            logger.warning(f"Mapping may be suboptimal: {result.message}")

        return np.round(result.x).astype(bool)

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

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)
        node_pairs = list(node_pair_sims)
        edge_pairs = list(edge_pair_sims)
        selected = (
            self.solve(
                x,
                y,
                node_pairs,
                edge_pairs,
                np.array(list(node_pair_sims.values()) + list(edge_pair_sims.values())),
            )
            if node_pairs
            else None
        )
        node_mapping = (
            frozendict[K, K]()
            if selected is None
            else self.node_mapping(node_pairs, selected[: len(node_pairs)])
        )

        # The edge variables drive the objective, but the mapping is derived from the
        # nodes so that it is maximal for the same reason as above.
        return self.similarity(
            x,
            y,
            node_mapping,
            self.induced_edge_mapping(x, y, node_mapping, edge_pair_sims),
            node_pair_sims,
            edge_pair_sims,
        )
