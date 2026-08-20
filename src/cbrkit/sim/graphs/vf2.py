import itertools
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, override

import numpy as np
from frozendict import frozendict
from scipy.optimize import linear_sum_assignment

from ...helpers import optional_dependencies, reverse_positional
from ...model.graph import Graph
from ...typing import SimFunc
from .common import BaseGraphSimFunc, ElementMatcher, GraphSim


def parallel_edges_match[E](
    edge_matcher: ElementMatcher[E],
    x_edges: Mapping[Any, Any],
    y_edges: Mapping[Any, Any],
) -> bool:
    """Whether every query edge can be paired with a distinct compatible case edge.

    Since the graph model allows parallel edges, the matchers of the multigraph
    backend receive all edges between a pair of nodes at once, and pairing them off
    injectively is a linear assignment problem.
    """
    if len(y_edges) > len(x_edges):
        return False

    # Without parallel edges there is nothing to pair off, and this callback sits in
    # the innermost loop of the matcher, so the assignment is worth skipping.
    if len(y_edges) == 1 and len(x_edges) == 1:
        return edge_matcher(
            next(iter(x_edges.values()))["value"],
            next(iter(y_edges.values()))["value"],
        )

    allowed = np.array(
        [
            [
                1.0 if edge_matcher(x_data["value"], y_data["value"]) else 0.0
                for x_data in x_edges.values()
            ]
            for y_data in y_edges.values()
        ]
    )
    rows, cols = linear_sum_assignment(allowed, maximize=True)

    return bool(allowed[rows, cols].sum() == len(y_edges))


with optional_dependencies():
    import rustworkx
    from networkx.algorithms.isomorphism import MultiDiGraphMatcher

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
    induced: bool = False

    @abstractmethod
    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        """Returns all subgraph isomorphism node mappings between the two graphs."""
        ...

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_mappings: list[frozendict[K, K]] = []
        next_permutations: list[Graph[K, N, E, G]] = [y]
        # Removing nodes in a different order yields the same subgraph, so the visited
        # node sets are tracked to keep the fallback exponential instead of factorial.
        visited: set[frozenset[K]] = {frozenset(y.nodes.keys())}

        while next_permutations and not node_mappings:
            current_permutations = next_permutations
            next_permutations = []

            for current_permutation in current_permutations:
                node_mappings.extend(self.node_mappings(x, current_permutation))

                if not self.maximum_common_subgraph:
                    continue

                # remove nodes from y to determine partial mappings
                for node_key in current_permutation.nodes:
                    remaining = frozenset(current_permutation.nodes.keys() - {node_key})

                    if remaining in visited:
                        continue

                    visited.add(remaining)
                    next_permutations.append(
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
                    )

        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)
        graph_sims: list[GraphSim[K]] = []

        for node_mapping in node_mappings:
            edge_mapping = self.induced_edge_mapping(x, y, node_mapping, edge_pair_sims)
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
class vf2_rustworkx[K, N, E, G](VF2Base[K, N, E, G]):
    """Graph similarity using the VF2 algorithm via rustworkx."""

    id_order: bool = False
    call_limit: int | None = None

    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        """Finds subgraph isomorphism node mappings using rustworkx."""
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
class vf2_networkx[K, N, E, G](VF2Base[K, N, E, G]):
    """Graph similarity using the VF2 algorithm via NetworkX."""

    def node_mappings(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        """Finds subgraph isomorphism node mappings using NetworkX."""
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
        graph_matcher = MultiDiGraphMatcher(
            larger_graph,
            smaller_graph,
            node_match=lambda x, y: node_matcher(x["value"], y["value"]),
            edge_match=lambda x, y: parallel_edges_match(edge_matcher, x, y),
        )

        mappings_iter = (
            graph_matcher.subgraph_isomorphisms_iter()
            if self.induced
            else graph_matcher.subgraph_monomorphisms_iter()
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
