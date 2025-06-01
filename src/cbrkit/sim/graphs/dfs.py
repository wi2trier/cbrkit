import itertools
from dataclasses import dataclass

from frozendict import frozendict

from ...helpers import get_logger, optional_dependencies
from ...model.graph import Graph, NetworkxEdge, NetworkxNode
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

logger = get_logger(__name__)

__all__ = ["dfs"]


with optional_dependencies():
    import networkx as nx

    from ...model.graph import to_networkx

    @dataclass(slots=True)
    class dfs[K, N, E, G](
        BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
    ):
        max_iterations: int = 0

        def __call__(
            self,
            x: Graph[K, N, E, G],
            y: Graph[K, N, E, G],
        ) -> GraphSim[K]:
            x_nx = to_networkx(x)
            y_nx = to_networkx(y)

            y_edges_lookup = {
                (e.source.key, e.target.key): e.key for e in y.edges.values()
            }
            x_edges_lookup = {
                (e.source.key, e.target.key): e.key for e in x.edges.values()
            }

            node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

            def node_subst_cost(y: NetworkxNode[K, N], x: NetworkxNode[K, N]) -> float:
                if sim := node_pair_sims.get((y["key"], x["key"])):
                    return 1.0 - sim

                return float("inf")

            def edge_subst_cost(
                y: NetworkxEdge[K, N, E], x: NetworkxEdge[K, N, E]
            ) -> float:
                if sim := edge_pair_sims.get((y["key"], x["key"])):
                    return 1.0 - sim

                return float("inf")

            ged_iter = nx.optimize_edit_paths(
                y_nx,
                x_nx,
                node_subst_cost=node_subst_cost,
                edge_subst_cost=edge_subst_cost,
            )

            node_edit_path: list[tuple[K, K]] = []
            edge_edit_path: list[tuple[tuple[K, K], tuple[K, K]]] = []

            for idx in itertools.count():
                if self.max_iterations > 0 and idx >= self.max_iterations:
                    break

                try:
                    node_edit_path, edge_edit_path, _ = next(ged_iter)
                except StopIteration:
                    break

            node_mapping = frozendict(
                (y_key, x_key)
                for y_key, x_key in node_edit_path
                if x_key is not None and y_key is not None
            )
            edge_mapping = frozendict(
                (
                    y_edges_lookup[y_key],
                    x_edges_lookup[x_key],
                )
                for y_key, x_key in edge_edit_path
                if x_key is not None
                and y_key is not None
                and x_key in x_edges_lookup
                and y_key in y_edges_lookup
            )

            return self.similarity(
                x,
                y,
                node_mapping,
                edge_mapping,
                node_pair_sims,
                edge_pair_sims,
            )
