from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, override

from cbrkit.helpers import SimSeqWrapper
from cbrkit.typing import (
    AggregatorFunc,
    AnySimFunc,
    Float,
    SimPairFunc,
    SimSeqFunc,
    SupportsMetadata,
)

from . import _model as model
from ._model import DataSimWrapper, Graph, GraphSim, Node


@dataclass(slots=True)
class isomorphism[K, N, E, G](
    SimPairFunc[Graph[K, N, E, G], GraphSim[K]],
    SupportsMetadata,
):
    """Compute subgraph isomorphisms between two graphs.

    - Convert the input graphs to Rustworkx graphs.
    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the similarity based on the node mapping.
    - Return the isomorphism mapping with the highest similarity.
    """

    node_matcher: Callable[[N, N], bool]
    edge_matcher: Callable[[E, E], bool]
    node_sim_func: SimSeqFunc[Node[K, N], Float]
    aggregator: AggregatorFunc[Any, Float]

    def __init__(
        self,
        node_matcher: Callable[[N, N], bool],
        edge_matcher: Callable[[E, E], bool],
        aggregator: AggregatorFunc[Any, Float],
        node_obj_sim: AnySimFunc[Node[K, N], Float] | None = None,
        node_data_sim: AnySimFunc[N, Float] | None = None,
    ) -> None:
        # verify that only one of the object or data similarity functions is provided
        if node_obj_sim and node_data_sim:
            raise ValueError(
                "Only one of the object or data similarity functions can be provided for nodes"
            )

        if node_data_sim:
            self.node_sim_func = DataSimWrapper(node_data_sim)
        elif node_obj_sim:
            self.node_sim_func = SimSeqWrapper(node_obj_sim)
        else:
            raise ValueError("Either node_obj_sim or node_data_sim must be provided")

        self.node_matcher = node_matcher
        self.edge_matcher = edge_matcher
        self.aggregator = aggregator

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        import rustworkx

        x_rw, x_lookup = model.to_rustworkx_with_lookup(x)
        y_rw, y_lookup = model.to_rustworkx_with_lookup(y)

        rw_mappings = rustworkx.vf2_mapping(
            y_rw,
            x_rw,
            subgraph=True,
            node_matcher=self.node_matcher,
            edge_matcher=self.edge_matcher,
        )

        node_mappings: list[dict[K, K]] = [
            {y_lookup[y_key]: x_lookup[x_key] for y_key, x_key in mapping.items()}
            for mapping in rw_mappings
        ]

        if len(node_mappings) == 0:
            return GraphSim(0.0, node_mappings={}, edge_mappings={})

        mapping_similarities: list[float] = []

        for node_mapping in node_mappings:
            node_pairs = [
                (x.nodes[x_key], y.nodes[y_key])
                for y_key, x_key in node_mapping.items()
            ]
            node_similarities = self.node_sim_func(node_pairs)
            mapping_similarities.append(self.aggregator(node_similarities))

        best_mapping_id, best_sim = max(
            enumerate(mapping_similarities),
            key=lambda x: x[1],
        )
        best_mapping = node_mappings[best_mapping_id]

        return GraphSim(best_sim, node_mappings=best_mapping, edge_mappings={})
