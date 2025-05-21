import itertools
from dataclasses import dataclass

from ...helpers import (
    batchify_sim,
    get_logger,
    optional_dependencies,
    unbatchify_sim,
    unpack_float,
)
from ...model.graph import (
    Edge,
    Graph,
    NetworkxEdge,
    NetworkxNode,
    Node,
    to_networkx,
)
from ...typing import AnySimFunc, Float, SimFunc
from ..wrappers import transpose_value
from .common import ElementMatcher, GraphSim, default_edge_sim, default_element_matcher

logger = get_logger(__name__)

__all__ = ["ged"]


with optional_dependencies():
    import networkx as nx

    @dataclass(slots=True, init=False)
    class ged[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
        node_sim_func: SimFunc[Node[K, N], Float]
        edge_sim_func: SimFunc[Edge[K, N, E], Float]
        node_matcher: ElementMatcher[N]
        edge_matcher: ElementMatcher[E]
        max_iterations: int

        def __init__(
            self,
            node_sim_func: AnySimFunc[N, Float],
            node_matcher: ElementMatcher[N],
            edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
            edge_matcher: ElementMatcher[E] = default_element_matcher,
            max_iterations: int = 0,
        ) -> None:
            self.max_iterations = max_iterations
            self.node_matcher = node_matcher
            self.edge_matcher = edge_matcher

            transposed_node_sim_func = transpose_value(node_sim_func)
            self.node_sim_func = unbatchify_sim(transposed_node_sim_func)
            self.edge_sim_func = unbatchify_sim(
                default_edge_sim(batchify_sim(transposed_node_sim_func))
                if edge_sim_func is None
                else edge_sim_func
            )

        def node_subst_cost(
            self, x: NetworkxNode[K, N], y: NetworkxNode[K, N]
        ) -> float:
            if self.node_matcher(x["value"], y["value"]):
                return 1.0 - unpack_float(self.node_sim_func(x["obj"], y["obj"]))

            return float("inf")

        def edge_subst_cost(
            self, x: NetworkxEdge[K, N, E], y: NetworkxEdge[K, N, E]
        ) -> float:
            if self.edge_matcher(x["value"], y["value"]):
                return 1.0 - unpack_float(self.edge_sim_func(x["obj"], y["obj"]))

            return float("inf")

        def __call__(
            self,
            x: Graph[K, N, E, G],
            y: Graph[K, N, E, G],
        ) -> GraphSim[K]:
            x_nx = to_networkx(x)
            y_nx = to_networkx(y)

            ged_iter = nx.optimize_graph_edit_distance(
                y_nx,
                x_nx,
                node_subst_cost=self.node_subst_cost,
                edge_subst_cost=self.edge_subst_cost,
            )

            node_edit_path: list[tuple[K, K]] = []
            edge_edit_path: list[tuple[tuple[K, K], tuple[K, K]]] = []
            cost: float = float("inf")

            for idx in itertools.count():
                if self.max_iterations > 0 and idx >= self.max_iterations:
                    break

                try:
                    node_edit_path, edge_edit_path, cost = next(ged_iter)
                except StopIteration:
                    break

            return GraphSim(
                1 - cost,
                {y_key: x_key for x_key, y_key in node_edit_path},
                {},
                {},
                {},
            )
