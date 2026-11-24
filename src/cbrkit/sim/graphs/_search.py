import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol, cast, override

from ..._util import sorted_keys
from ...helpers import batchify, frozendict
from ...model.graph import Graph, GraphElementType
from ...model.similarity import GraphSim
from ...schema import Family, serializable
from .._nested import NestedMeasure
from ._base import (
    BaseGraphSimFunc,
    PairSimilarities,
    _induced_edge_candidates,
    _mutually_unique_mapping,
)
from ._matchers import ElementMatcher

__all__ = [
    "SEARCH_INITS",
    "InitEmpty",
    "InitFromMeasure",
    "InitUniqueMatches",
    "SearchGraphSimFunc",
    "SearchState",
    "SearchStateInit",
    "next_elem",
]


@dataclass(slots=True, frozen=True)
class SearchState[K: Hashable]:
    """Current state of a graph search with node and edge mappings.

    Args:
        node_mapping: Case node keys by query node key.
        edge_mapping: Case edge keys by query edge key.
        open_y_nodes: Query nodes that are not mapped yet, also those without a legal counterpart.
        open_y_edges: Query edges that are not mapped yet, also those without a legal counterpart.
        open_x_nodes: Case nodes that are not mapped yet.
        open_x_edges: Case edges that are not mapped yet.
    """

    # mappings are from y/query to x/case
    node_mapping: frozendict[K, K]
    edge_mapping: frozendict[K, K]
    # contains all elements from the query that are not yet mapped
    # can be different from mapping.keys() if no candidate in x/case exists
    open_y_nodes: frozenset[K]
    open_y_edges: frozenset[K]
    # contains all elements from the case that are not yet mapped
    # must be identical to mapping.values() but is stored to optimize lookup
    open_x_nodes: frozenset[K]
    open_x_edges: frozenset[K]


def next_elem[K: Hashable](
    elements: Collection[K],
    key: Callable[[K], Any] | None = None,
) -> K:
    """Select the next element from a set deterministically.

    Returns the smallest element, compared by their representation if they are not
    orderable.

    Args:
        elements: Set of elements to choose from

    Returns:
        A single element from the set

    Raises:
        ValueError: If the set is empty
    """
    if not elements:
        raise ValueError("Cannot select from empty set")

    if len(elements) == 1:
        return next(iter(elements))

    try:
        return min(cast(Iterable[Any], elements), key=key)
    except TypeError:
        return min(elements, key=lambda elem: repr(key(elem) if key else elem))


class SearchStateInit[K: Hashable, N, E, G](Protocol):
    """Initializes the search states of many graph pairs at once, each given as a case and a query."""

    def __call__(
        self,
        batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
        /,
    ) -> Sequence[SearchState[K]]: ...


def _state[K: Hashable, N, E, G](
    case: Graph[K, N, E, G],
    query: Graph[K, N, E, G],
    node_mapping: frozendict[K, K],
    edge_mapping: frozendict[K, K],
) -> SearchState[K]:
    """The state that starts from the given mappings."""
    return SearchState(
        node_mapping,
        edge_mapping,
        frozenset(query.nodes.keys() - node_mapping.keys()),
        frozenset(query.edges.keys() - edge_mapping.keys()),
        frozenset(case.nodes.keys() - node_mapping.values()),
        frozenset(case.edges.keys() - edge_mapping.values()),
    )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class InitEmpty[K: Hashable, N, E, G](SearchStateInit[K, N, E, G]):
    """Initializes search with an empty mapping."""

    kind: Literal["empty"] = field(default="empty", repr=False)

    def __call__(
        self,
        batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> list[SearchState[K]]:
        return [_state(x, y, frozendict(), frozendict()) for x, y in batches]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class InitUniqueMatches[K: Hashable, N, E, G](SearchStateInit[K, N, E, G]):
    """Initializes search by pre-mapping uniquely matchable nodes and edges."""

    kind: Literal["unique_matches"] = field(default="unique_matches", repr=False)

    def __call__(
        self,
        batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> list[SearchState[K]]:
        return [self.unique(x, y, node_matcher, edge_matcher) for x, y in batches]

    @staticmethod
    def unique(
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> SearchState[K]:
        """The state of a single pair, which maps the elements with a single legal counterpart."""
        node_mapping = _mutually_unique_mapping(
            (y_key, x_key)
            for y_key, x_key in itertools.product(query.nodes.keys(), case.nodes.keys())
            if node_matcher(case.nodes[x_key].value, query.nodes[y_key].value)
        )
        edge_mapping = _mutually_unique_mapping(
            _induced_edge_candidates(case, query, node_mapping, edge_matcher)
        )

        return _state(case, query, node_mapping, edge_mapping)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class InitFromMeasure[K: Hashable, N, E, G](SearchStateInit[K, N, E, G]):
    """Initializes search with the mapping of another graph measure, such as a fast `LAP`, which maps all pairs in one batch.

    Args:
        measure: The graph measure whose mapping the search starts from.
    """

    measure: NestedMeasure[Graph[K, N, E, G], GraphSim[K]]
    kind: Literal["from_measure"] = field(default="from_measure", repr=False)

    def __call__(
        self,
        batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]],
        node_matcher: ElementMatcher[N],
        edge_matcher: ElementMatcher[E],
    ) -> list[SearchState[K]]:
        return [
            _state(x, y, sim.mapping, sim.edge_mapping)
            for (x, y), sim in zip(
                batches, batchify(self.measure)(batches), strict=True
            )
        ]


SEARCH_INITS = Family(
    "search initialization",
    (InitEmpty, InitUniqueMatches, InitFromMeasure),
)


@dataclass(slots=True, frozen=True, kw_only=True)
class SearchGraphSimFunc[K: Hashable, N, E, G](BaseGraphSimFunc[K, N, E, G], ABC):
    """Graph similarity function using search-based node and edge mapping.

    Args:
        init: A function to initialize the state.
        partial_mapping: Also consider leaving an element of the query unmapped when
            it has a legal counterpart in the case.
            [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) allow
            this, and it is needed whenever declining a mapping frees an element of
            the case for a more valuable use elsewhere.
            Disabling it prunes one successor per expansion, which shrinks the search
            space by roughly a factor of two but gives up the guarantee of finding the
            best mapping.
            Disabled by default.
    """

    init: Annotated[SearchStateInit[K, N, E, G], SEARCH_INITS] = field(
        default_factory=InitUniqueMatches
    )
    partial_mapping: bool = False

    @override
    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]], /
    ) -> list[GraphSim[K]]:
        states = self.init(batches, self.node_matcher, self.edge_matcher)

        return [
            self.search(x, y, sims, state)
            for (x, y), sims, state in zip(
                batches, self.batch_pair_similarities(batches), states, strict=True
            )
        ]

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        return self.search(case, query, sims, self.init_search_state(case, query))

    @abstractmethod
    def search(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
        state: SearchState[K],
    ) -> GraphSim[K]:
        """Maps a single graph pair by searching from the given state."""
        ...

    def init_search_state(
        self, case: Graph[K, N, E, G], query: Graph[K, N, E, G]
    ) -> SearchState[K]:
        """Initializes the search state of a single pair using the configured init function."""
        return self.init([(case, query)], self.node_matcher, self.edge_matcher)[0]

    def finished(self, state: SearchState[K]) -> bool:
        """Returns True when all query nodes and edges have been processed."""
        # the following condition could save a few iterations, but needs to be tested
        # return (not state.open_y_nodes and not state.open_y_edges) or (
        #     not state.open_x_nodes and not state.open_x_edges
        # )
        return not state.open_y_nodes and not state.open_y_edges

    def legal_node_mapping(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        """Checks whether mapping a query node to a case node is legal."""
        return (
            self.node_matcher(case.nodes[x_key].value, query.nodes[y_key].value)
            and y_key in state.open_y_nodes
            and x_key in state.open_x_nodes
        )

    def legal_edge_mapping(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        x_key: K,
        y_key: K,
    ) -> bool:
        """Checks whether mapping a query edge to a case edge is legal."""
        x_value = case.edges[x_key]
        y_value = query.edges[y_key]

        return (
            self.edge_matcher(x_value.value, y_value.value)
            and y_key in state.open_y_edges
            and x_key in state.open_x_edges
            # source and target of the edge must be mapped to the same nodes
            and x_value.source.key == state.node_mapping.get(y_value.source.key)
            and x_value.target.key == state.node_mapping.get(y_value.target.key)
        )

    def append_skipped(
        self,
        next_states: list[SearchState[K]],
        state: SearchState[K],
        y_key: K,
        elem_type: GraphElementType,
    ) -> list[SearchState[K]]:
        """Adds the successor that leaves the selected query element unmapped.

        It is the only successor whenever nothing legal is available, and
        `partial_mapping` decides whether it is offered next to the legal ones too.
        """
        if self.partial_mapping or not next_states:
            is_node = elem_type == "node"
            next_states.append(
                SearchState(
                    state.node_mapping,
                    state.edge_mapping,
                    state.open_y_nodes - {y_key} if is_node else state.open_y_nodes,
                    state.open_y_edges if is_node else state.open_y_edges - {y_key},
                    state.open_x_nodes,
                    state.open_x_edges,
                )
            )

        return next_states

    def expand_node(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:
        """Generates successor states by mapping a query node to all legal case nodes.

        Unless `partial_mapping` is disabled, leaving the node unmapped is always
        offered as a successor.
        Otherwise a case node could be consumed by a node of the query that has no
        alternative, blocking a more valuable mapping elsewhere.
        """
        next_states: list[SearchState[K]] = [
            SearchState(
                state.node_mapping | {y_key: x_key},
                state.edge_mapping,
                state.open_y_nodes - {y_key},
                state.open_y_edges,
                state.open_x_nodes - {x_key},
                state.open_x_edges,
            )
            for x_key in sorted_keys(state.open_x_nodes)
            if self.legal_node_mapping(case, query, state, x_key, y_key)
        ]

        return self.append_skipped(next_states, state, y_key, "node")

    def expand_edge(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        state: SearchState[K],
        y_key: K,
    ) -> list[SearchState[K]]:
        """Generates successor states by mapping a query edge to all legal case edges.

        As for nodes, leaving the edge unmapped is offered as a successor unless
        `partial_mapping` is disabled.
        """
        next_states: list[SearchState[K]] = [
            SearchState(
                state.node_mapping,
                state.edge_mapping | {y_key: x_key},
                state.open_y_nodes,
                state.open_y_edges - {y_key},
                state.open_x_nodes,
                state.open_x_edges - {x_key},
            )
            for x_key in sorted_keys(state.open_x_edges)
            if self.legal_edge_mapping(case, query, state, x_key, y_key)
        ]

        return self.append_skipped(next_states, state, y_key, "edge")
