import dataclasses
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
    - For each isomorphism, compute the global similarity.
    - Return the isomorphism mapping with the highest similarity.
    """

    id_order: bool = False
    subgraph: bool = True
    induced: bool = False
    call_limit: int | None = None
    max_iterations: int = 0

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
            self_inv = dataclasses.replace(self, _invert=True)
            return self.invert_similarity(x, y, self_inv(x=y, y=x))

        y_rx, y_lookup = to_rustworkx_with_lookup(y)
        x_rx, x_lookup = to_rustworkx_with_lookup(x)

        # Checks if there is a subgraph of `first` isomorphic to `second`.
        # Returns an iterator over dictionaries of node indices from `first`
        # to node indices in `second` representing the mapping found.
        # As such, `first` must be the larger graph and `second` the smaller one.
        mappings_iter = rustworkx.vf2_mapping(
            x_rx,
            y_rx,
            node_matcher=self.node_matcher,
            edge_matcher=self.edge_matcher,
            id_order=self.id_order,
            subgraph=self.subgraph,
            induced=self.induced,
            call_limit=self.call_limit,
        )

        node_mappings: list[frozendict[K, K]] = []
        graph_sims: list[GraphSim[K]] = []

        for idx in itertools.count():
            if self.max_iterations > 0 and idx >= self.max_iterations:
                break

            try:
                node_mappings.append(
                    frozendict(
                        (y_lookup[y_idx], x_lookup[x_idx])
                        for x_idx, y_idx in next(mappings_iter).items()
                    )
                )
            except StopIteration:
                break

        for node_mapping in node_mappings:
            edge_mapping = self.induced_edge_mapping(x, y, node_mapping)
            node_pair_sims, edge_pair_sims = self.pair_similarities(
                x, y, list(node_mapping.items()), list(edge_mapping.items())
            )
            graph_sims.append(
                self.similarity(
                    x,
                    y,
                    node_mapping,
                    edge_mapping,
                    node_pair_sims,
                    edge_pair_sims,
                )
            )

        return max(
            graph_sims,
            key=lambda sim: sim.value,
            default=GraphSim(
                0.0, frozendict(), frozendict(), frozendict(), frozendict()
            ),
        )
