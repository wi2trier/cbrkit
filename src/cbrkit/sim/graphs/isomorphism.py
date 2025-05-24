import itertools
from dataclasses import dataclass
from typing import override

from frozendict import frozendict

from ...helpers import optional_dependencies
from ...model.graph import Graph
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

with optional_dependencies():
    import rustworkx

    from ...model.graph import to_rustworkx_with_lookup


@dataclass(slots=True)
class isomorphism[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Compute subgraph isomorphisms between two graphs.

    - Convert the input graphs to Rustworkx graphs.
    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the similarity based on the node mapping.
    - Return the isomorphism mapping with the highest similarity.
    """

    id_order: bool = True
    subgraph: bool = True
    induced: bool = True
    call_limit: int | None = None
    max_iterations: int = 0

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        x_rw, x_lookup = to_rustworkx_with_lookup(x)
        y_rw, y_lookup = to_rustworkx_with_lookup(y)

        mappings_iter = rustworkx.vf2_mapping(
            y_rw,
            x_rw,
            node_matcher=self.node_matcher,
            edge_matcher=self.edge_matcher,
            id_order=self.id_order,
            subgraph=self.subgraph,
            induced=self.induced,
            call_limit=self.call_limit,
        )

        node_mappings: list[dict[K, K]] = []
        graph_sims: list[GraphSim[K]] = []

        for idx in itertools.count():
            if self.max_iterations > 0 and idx >= self.max_iterations:
                break

            try:
                node_mappings.append(
                    {
                        y_lookup[y_key]: x_lookup[x_key]
                        for y_key, x_key in next(mappings_iter).items()
                    }
                )
            except StopIteration:
                break

        for node_mapping in node_mappings:
            node_pair_sims = self.node_pair_similarities(
                x, y, list(node_mapping.items())
            )
            graph_sims.append(
                self.similarity(
                    x,
                    y,
                    frozendict(node_mapping),
                    frozendict(),
                    node_pair_sims,
                    frozendict(),
                )
            )

        if len(graph_sims) == 0:
            return GraphSim(0.0, frozendict(), frozendict(), frozendict(), frozendict())

        return max(graph_sims, key=lambda sim: sim.value)
