import itertools
from dataclasses import dataclass
from typing import Protocol, cast

from frozendict import frozendict

from ...helpers import get_logger, optional_dependencies
from ...model.graph import Graph, NetworkxEdge, NetworkxNode
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)


class RootsFunc[K, N, E, G](Protocol):
    """Support for matching rooted graphs

    Returns:
        Tuple where the first element is a node in y and the second is a node in x.
    """

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> tuple[K, K]: ...


with optional_dependencies():
    import networkx as nx

    from ...model.graph import to_networkx

    @dataclass(slots=True)
    class dfs[K, N, E, G](
        BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
    ):
        """Graph similarity using depth-first search over node and edge mappings."""

        max_iterations: int = 0
        upper_bound: float | None = None
        strictly_decreasing: bool = True
        timeout: float | None = None
        roots_func: RootsFunc[K, N, E, G] | None = None

        def __call__(
            self,
            x: Graph[K, N, E, G],
            y: Graph[K, N, E, G],
        ) -> GraphSim[K]:
            x_nx = to_networkx(x)
            y_nx = to_networkx(y)

            node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

            def node_subst_cost(y: NetworkxNode[K, N], x: NetworkxNode[K, N]) -> float:
                """Returns the substitution cost for a node pair as one minus similarity."""
                # Compared against None so that a legitimate similarity of zero is not
                # mistaken for an illegal pair.
                sim = node_pair_sims.get((y["key"], x["key"]))

                return 1.0 - sim if sim is not None else float("inf")

            def edge_subst_cost(
                y: NetworkxEdge[K, N, E], x: NetworkxEdge[K, N, E]
            ) -> float:
                """Returns the substitution cost for an edge pair as one minus similarity."""
                sim = edge_pair_sims.get((y["key"], x["key"]))

                return 1.0 - sim if sim is not None else float("inf")

            ged_iter = nx.optimize_edit_paths(
                y_nx,
                x_nx,
                node_subst_cost=node_subst_cost,
                edge_subst_cost=edge_subst_cost,
                node_del_cost=lambda y: self.node_del_cost,
                node_ins_cost=lambda x: self.node_ins_cost,
                edge_del_cost=lambda y: self.edge_del_cost,
                edge_ins_cost=lambda x: self.edge_ins_cost,
                upper_bound=self.upper_bound,
                strictly_decreasing=self.strictly_decreasing,
                timeout=self.timeout,
                roots=self.roots_func(x, y) if self.roots_func else None,
            )

            node_edit_path: list[tuple[K | None, K | None]] = []
            # Edges of a multigraph are identified by source, target and edge key.
            edge_edit_path: list[
                tuple[tuple[K, K, K] | None, tuple[K, K, K] | None]
            ] = []

            for idx in itertools.count():
                if self.max_iterations > 0 and idx >= self.max_iterations:
                    break

                try:
                    node_edit_path, edge_edit_path, _ = next(ged_iter)
                except StopIteration:
                    break

            node_mapping = cast(
                frozendict[K, K],
                frozendict(
                    (y_key, x_key)
                    for y_key, x_key in node_edit_path
                    if x_key is not None and y_key is not None
                ),
            )
            # Networkx identifies the edges of a multigraph by its own key, so the
            # key of the original edge is read back from the edge data.
            edge_mapping = frozendict(
                (y_nx.edges[y_edge]["key"], x_nx.edges[x_edge]["key"])
                for y_edge, x_edge in edge_edit_path
                if x_edge is not None and y_edge is not None
            )

            return self.similarity(
                x,
                y,
                node_mapping,
                edge_mapping,
                node_pair_sims,
                edge_pair_sims,
            )
