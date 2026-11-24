import itertools
import logging
import math
from collections import deque
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

from pydantic import Field

from ..._lazy import extra_hint
from ...helpers import frozendict
from ...model.graph import Graph, NetworkxEdge, NetworkxNode
from ...model.similarity import GraphSim
from ...schema import PythonOnly, serializable
from ._base import BaseGraphSimFunc, PairSimilarities

logger = logging.getLogger(__name__)


type RootsFunc[K: Hashable, N, E, G] = Callable[
    [Graph[K, N, E, G], Graph[K, N, E, G]], tuple[K, K]
]
"""Roots of rooted graphs as a pair of a query node and a case node, given the case and the query."""


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class DFS[K: Hashable, N, E, G](BaseGraphSimFunc[K, N, E, G]):
    """Graph similarity using depth-first search over node and edge mappings.

    Based on the graph edit path search of NetworkX (`optimize_edit_paths`).

    Args:
        max_iterations: Maximal number of improved edit paths to consider, unbounded by default.
        upper_bound: Maximal edit cost to consider.
        strictly_decreasing: Whether only edit paths with a lower cost than the previous are returned.
        timeout: Maximal search time in seconds, unbounded by default.
        roots: Pair of query and case nodes that have to be mapped onto each other.
    """

    max_iterations: Annotated[int, Field(ge=0)] | None = None
    upper_bound: float | None = None
    strictly_decreasing: bool = True
    timeout: Annotated[float, Field(ge=0)] | None = None
    roots: PythonOnly[RootsFunc[K, N, E, G] | None] = None
    kind: Literal["dfs"] = field(default="dfs", repr=False)

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        with extra_hint("cbrkit.sim.graphs.DFS"):
            import networkx as nx

            from ...model.graph._networkx import to_networkx

        x_nx = to_networkx(case)
        y_nx = to_networkx(query)
        node_pair_sims, edge_pair_sims = sims.nodes, sims.edges

        def node_subst_cost(y: NetworkxNode[K, N], x: NetworkxNode[K, N]) -> float:
            """Returns the substitution cost for a node pair as one minus similarity."""
            # Compared against None so that a legitimate similarity of zero is not
            # mistaken for an illegal pair.
            sim = node_pair_sims.get((y["key"], x["key"]))

            return self.node_weight * (1.0 - sim) if sim is not None else math.inf

        def edge_subst_cost(
            y: NetworkxEdge[K, N, E], x: NetworkxEdge[K, N, E]
        ) -> float:
            """Returns the substitution cost for an edge pair as one minus similarity."""
            sim = edge_pair_sims.get((y["key"], x["key"]))

            return self.edge_weight * (1.0 - sim) if sim is not None else math.inf

        ged_iter = nx.optimize_edit_paths(
            y_nx,
            x_nx,
            node_subst_cost=node_subst_cost,
            edge_subst_cost=edge_subst_cost,
            node_del_cost=lambda y: self.node_deletion_cost,
            node_ins_cost=lambda x: self.node_insertion_cost,
            edge_del_cost=lambda y: self.edge_deletion_cost,
            edge_ins_cost=lambda x: self.edge_insertion_cost,
            upper_bound=self.upper_bound,
            strictly_decreasing=self.strictly_decreasing,
            timeout=self.timeout,
            roots=self.roots(case, query) if self.roots else None,
        )

        # Every edit path improves on the previous one, so the last one is the best.
        paths = deque(itertools.islice(ged_iter, self.max_iterations), maxlen=1)
        node_edit_path: list[tuple[K | None, K | None]] = paths[0][0] if paths else []

        node_mapping: frozendict[K, K] = frozendict(
            (y_key, x_key)
            for y_key, x_key in node_edit_path
            if x_key is not None and y_key is not None
        )

        # The edit path of networkx does not tell parallel edges apart, so the
        # edges are induced by the nodes as for every other measure.
        return self.similarity(case, query, node_mapping, sims)
