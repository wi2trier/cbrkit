import itertools
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, override

import numpy as np
from pydantic import Field
from scipy.optimize import linear_sum_assignment

from ..._lazy import extra_hint
from ..._util import sorted_keys
from ...helpers import frozendict
from ...model.graph import Graph
from ...model.similarity import GraphSim
from ...schema import serializable
from ._base import BaseGraphSimFunc, PairSimilarities
from ._matchers import ElementMatcher


@dataclass(slots=True, frozen=True)
class _Swapped[T]:
    """Matcher with case and query swapped, for when the query is the larger graph."""

    matcher: ElementMatcher[T]

    def __call__(self, case: T, query: T) -> bool:
        return self.matcher(query, case)


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


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseVF2[K: Hashable, N, E, G](BaseGraphSimFunc[K, N, E, G], ABC):
    """Compute subgraph isomorphisms between two graphs.

    - Compute all possible subgraph isomorphisms between the two graphs.
    - For each isomorphism, compute the global similarity.
    - Return the isomorphism mapping with the highest similarity.

    Args:
        max_iterations: Maximal number of isomorphisms to consider, unbounded by default.
        maximum_common_subgraph: Whether to remove query nodes until an isomorphism is found.
        induced: Whether the subgraphs have to be induced.
    """

    max_iterations: Annotated[int, Field(ge=0)] | None = None
    maximum_common_subgraph: bool = True
    induced: bool = False

    @abstractmethod
    def isomorphisms(
        self,
        larger: Graph[K, N, E, G],
        smaller: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> Iterator[Mapping[K, K]]:
        """Enumerates the subgraph isomorphisms as node keys of the larger graph to those of the smaller one."""
        ...

    def node_mappings(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
    ) -> list[frozendict[K, K]]:
        """Returns all subgraph isomorphism node mappings between the two graphs."""
        # The backends need the larger graph first, so the case and the query swap
        # if the query is larger.
        swapped = len(query.nodes) + len(query.edges) > len(case.nodes) + len(
            case.edges
        )
        mappings = (
            self.isomorphisms(
                query, case, _Swapped(self.node_matcher), _Swapped(self.edge_matcher)
            )
            if swapped
            else self.isomorphisms(case, query, self.node_matcher, self.edge_matcher)
        )

        # The mappings go from the larger to the smaller graph, so they are
        # inverted unless the query is the larger one.
        return [
            frozendict(
                mapping.items()
                if swapped
                else ((smaller, larger) for larger, smaller in mapping.items())
            )
            for mapping in itertools.islice(mappings, self.max_iterations)
        ]

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        node_mappings: list[frozendict[K, K]] = []
        next_permutations: list[Graph[K, N, E, G]] = [query]
        # Removing nodes in a different order yields the same subgraph, so the visited
        # node sets are tracked to keep the fallback exponential instead of factorial.
        visited: set[frozenset[K]] = {frozenset(query.nodes.keys())}

        while next_permutations and not node_mappings:
            current_permutations = next_permutations
            next_permutations = []

            for current_permutation in current_permutations:
                node_mappings.extend(self.node_mappings(case, current_permutation))

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

        # Backends may enumerate in hash order, so ties are broken by the mapping.
        graph_sims = [
            self.similarity(case, query, node_mapping, sims)
            for node_mapping in sorted(
                node_mappings, key=lambda mapping: repr(sorted_keys(mapping.items()))
            )
        ]

        return max(
            graph_sims,
            key=lambda sim: sim.value,
            default=GraphSim(
                value=0.0,
                similarities=frozendict(),
                mapping=frozendict(),
                edge_similarities=frozendict(),
                edge_mapping=frozendict(),
            ),
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class VF2Rustworkx[K: Hashable, N, E, G](BaseVF2[K, N, E, G]):
    """Graph similarity using the VF2 algorithm via rustworkx.

    Args:
        id_order: Whether rustworkx visits the nodes in the order of their ids.
        max_expansions: Maximal number of states that rustworkx visits, unbounded by default.
    """

    id_order: bool = False
    max_expansions: Annotated[int, Field(ge=0)] | None = None
    kind: Literal["vf2_rustworkx"] = field(default="vf2_rustworkx", repr=False)

    @override
    def isomorphisms(
        self,
        larger: Graph[K, N, E, G],
        smaller: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> Iterator[Mapping[K, K]]:
        with extra_hint("cbrkit.sim.graphs.VF2Rustworkx"):
            import rustworkx

            from ...model.graph._rustworkx import to_rustworkx_with_lookup

        larger_graph, larger_lookup = to_rustworkx_with_lookup(larger)
        smaller_graph, smaller_lookup = to_rustworkx_with_lookup(smaller)
        mappings = rustworkx.vf2_mapping(
            larger_graph,
            smaller_graph,
            node_matcher=node_matcher,
            edge_matcher=edge_matcher,
            subgraph=True,
            id_order=self.id_order,
            induced=self.induced,
            call_limit=self.max_expansions,
        )

        # rustworkx maps node indices, which are translated back to keys
        return (
            {
                larger_lookup[larger_index]: smaller_lookup[smaller_index]
                for larger_index, smaller_index in mapping.items()
            }
            for mapping in mappings
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class VF2Networkx[K: Hashable, N, E, G](BaseVF2[K, N, E, G]):
    """Graph similarity using the VF2 algorithm via NetworkX."""

    kind: Literal["vf2_networkx"] = field(default="vf2_networkx", repr=False)

    @override
    def isomorphisms(
        self,
        larger: Graph[K, N, E, G],
        smaller: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> Iterator[Mapping[K, K]]:
        with extra_hint("cbrkit.sim.graphs.VF2Networkx"):
            from networkx.algorithms.isomorphism import MultiDiGraphMatcher

            from ...model.graph._networkx import to_networkx

        graph_matcher = MultiDiGraphMatcher(
            to_networkx(larger),
            to_networkx(smaller),
            node_match=lambda x, y: node_matcher(x["value"], y["value"]),
            edge_match=lambda x, y: parallel_edges_match(edge_matcher, x, y),
        )

        return (
            graph_matcher.subgraph_isomorphisms_iter()
            if self.induced
            else graph_matcher.subgraph_monomorphisms_iter()
        )
