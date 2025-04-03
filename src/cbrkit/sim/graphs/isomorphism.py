from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, override

from ...helpers import batchify_sim, optional_dependencies, unpack_float
from ...model.graph import Graph
from ...typing import (
    AggregatorFunc,
    AnySimFunc,
    Float,
    SimFunc,
)
from ..aggregator import default_aggregator
from ..wrappers import transpose_value
from .common import ElementMatcher, GraphSim, default_element_matcher

with optional_dependencies():
    import rustworkx

    from ...model.graph import to_rustworkx_with_lookup


@dataclass(slots=True, frozen=True)
class isomorphism[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """Compute subgraph isomorphisms between two graphs.

    - Convert the input graphs to Rustworkx graphs.
    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the similarity based on the node mapping.
    - Return the isomorphism mapping with the highest similarity.
    """

    node_sim_func: AnySimFunc[N, Float]
    node_matcher: ElementMatcher[N]
    edge_matcher: ElementMatcher[E] = default_element_matcher
    aggregator: AggregatorFunc[Any, Float] = default_aggregator
    id_order: bool = True
    subgraph: bool = True
    induced: bool = True
    call_limit: int | None = None

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_sim_func = batchify_sim(transpose_value(self.node_sim_func))

        x_rw, x_lookup = to_rustworkx_with_lookup(x)
        y_rw, y_lookup = to_rustworkx_with_lookup(y)

        rw_mappings = rustworkx.vf2_mapping(
            y_rw,
            x_rw,
            node_matcher=self.node_matcher,
            edge_matcher=self.edge_matcher,
            id_order=self.id_order,
            subgraph=self.subgraph,
            induced=self.induced,
            call_limit=self.call_limit,
        )

        node_mappings: list[dict[K, K]] = [
            {y_lookup[y_key]: x_lookup[x_key] for y_key, x_key in mapping.items()}
            for mapping in rw_mappings
        ]

        if len(node_mappings) == 0:
            return GraphSim(0.0, {}, {}, {}, {})

        global_sims: list[float] = []
        local_sims: list[Sequence[Float]] = []

        for node_mapping in node_mappings:
            node_pairs = [
                (x.nodes[x_key], y.nodes[y_key])
                for y_key, x_key in node_mapping.items()
            ]
            node_similarities = node_sim_func(node_pairs)
            local_sims.append(node_similarities)
            global_sims.append(self.aggregator(node_similarities))

        best_mapping_id, best_global_sim = max(
            enumerate(global_sims),
            key=lambda x: x[1],
        )
        best_mapping = node_mappings[best_mapping_id]
        best_local_sims = {
            y_key: unpack_float(local_sim)
            for y_key, local_sim in zip(
                best_mapping.keys(),
                local_sims[best_mapping_id],
                strict=True,
            )
        }

        return GraphSim(best_global_sim, best_mapping, {}, best_local_sims, {})
