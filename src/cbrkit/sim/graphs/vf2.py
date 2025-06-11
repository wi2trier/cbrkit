import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

from frozendict import frozendict

from ...helpers import optional_dependencies, reverse_positional
from ...model.graph import Graph
from ...typing import SimFunc
from .common import BaseGraphSimFunc, GraphSim

with optional_dependencies():
    import rustworkx
    from networkx.algorithms.isomorphism import DiGraphMatcher

    from ...model.graph import to_networkx, to_rustworkx_with_lookup


@dataclass(slots=True)
class VF2Base[K, N, E, G](
    ABC, BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Compute subgraph isomorphisms between two graphs.

    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the global similarity.
    - Return the isomorphism mapping with the highest similarity.
    """

    max_iterations: int = 0
    maximum_common_subgraph: bool = True

    @abstractmethod
    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]: ...

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_mappings: list[frozendict[K, K]] = []
        next_permutations: list[Graph] = [y]

        while next_permutations and not node_mappings:
            current_permutations = next_permutations
            next_permutations = []

            for current_permutation in current_permutations:
                node_mappings.extend(self.node_mappings(x, current_permutation))

                if self.maximum_common_subgraph:
                    # remove nodes from y to determine partial mappings
                    next_permutations.extend(
                        Graph(
                            nodes=frozendict(
                                (k, v)
                                for k, v in current_permutation.nodes.items()
                                if k != node_key
                            ),
                            edges=frozendict(
                                (k, v)
                                for k, v in current_permutation.edges.items()
                                if v.source.key != node_key and v.target.key != node_key
                            ),
                            value=current_permutation.value,
                        )
                        for node_key in current_permutation.nodes.keys()
                    )

        graph_sims: list[GraphSim[K]] = []

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
                0.0,
                frozendict(),
                frozendict(),
                frozendict(),
                frozendict(),
            ),
        )


@dataclass(slots=True)
class vf2_rustworkx[K, N, E, G](VF2Base):
    id_order: bool = False
    induced: bool = False
    call_limit: int | None = None

    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
            larger_graph, larger_graph_lookup = to_rustworkx_with_lookup(y)
            smaller_graph, smaller_graph_lookup = to_rustworkx_with_lookup(x)
            node_matcher = reverse_positional(self.node_matcher)
            edge_matcher = reverse_positional(self.edge_matcher)
        else:
            larger_graph, larger_graph_lookup = to_rustworkx_with_lookup(x)
            smaller_graph, smaller_graph_lookup = to_rustworkx_with_lookup(y)
            node_matcher = self.node_matcher
            edge_matcher = self.edge_matcher

        # Checks if there is a subgraph of `first` isomorphic to `second`.
        # Returns an iterator over dictionaries of node indices from `first`
        # to node indices in `second` representing the mapping found.
        # As such, `first` must be the larger graph and `second` the smaller one.
        mappings_iter = rustworkx.vf2_mapping(
            larger_graph,
            smaller_graph,
            node_matcher=node_matcher,
            edge_matcher=edge_matcher,
            subgraph=True,
            id_order=self.id_order,
            induced=self.induced,
            call_limit=self.call_limit,
        )

        node_mappings: list[frozendict[K, K]] = []

        for idx in itertools.count():
            if self.max_iterations > 0 and idx >= self.max_iterations:
                break

            try:
                if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
                    # y -> x (as needed)
                    node_mappings.append(
                        frozendict(
                            (
                                larger_graph_lookup[larger_idx],
                                smaller_graph_lookup[smaller_idx],
                            )
                            for larger_idx, smaller_idx in next(mappings_iter).items()
                        )
                    )
                else:
                    # x ->  y (needs to be inverted)
                    node_mappings.append(
                        frozendict(
                            (
                                smaller_graph_lookup[smaller_idx],
                                larger_graph_lookup[larger_idx],
                            )
                            for larger_idx, smaller_idx in next(mappings_iter).items()
                        )
                    )
            except StopIteration:
                break

        return node_mappings


@dataclass(slots=True)
class vf2_networkx[K, N, E, G](VF2Base):
    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
            larger_graph = to_networkx(y)
            smaller_graph = to_networkx(x)
            node_matcher = reverse_positional(self.node_matcher)
            edge_matcher = reverse_positional(self.edge_matcher)
        else:
            larger_graph = to_networkx(x)
            smaller_graph = to_networkx(y)
            node_matcher = self.node_matcher
            edge_matcher = self.edge_matcher

        # `first` must be the larger graph and `second` the smaller one.
        graph_matcher = DiGraphMatcher(
            larger_graph,
            smaller_graph,
            node_match=lambda x, y: node_matcher(x["value"], y["value"]),
            edge_match=lambda x, y: edge_matcher(x["value"], y["value"]),
        )

        mappings_iter = graph_matcher.subgraph_isomorphisms_iter()
        node_mappings: list[frozendict[K, K]] = []

        for idx in itertools.count():
            if self.max_iterations > 0 and idx >= self.max_iterations:
                break

            try:
                if len(y.nodes) + len(y.edges) > len(x.nodes) + len(x.edges):
                    # y -> x (as needed)
                    node_mappings.append(
                        frozendict(
                            (larger_idx, smaller_idx)
                            for larger_idx, smaller_idx in next(mappings_iter).items()
                        )
                    )
                else:
                    # x ->  y (needs to be inverted)
                    node_mappings.append(
                        frozendict(
                            (smaller_idx, larger_idx)
                            for larger_idx, smaller_idx in next(mappings_iter).items()
                        )
                    )
            except StopIteration:
                break

        return node_mappings


vf2 = vf2_rustworkx
