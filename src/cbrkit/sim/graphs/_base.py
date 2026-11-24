import itertools
import math
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Collection, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, override

import numpy as np
import scipy

from ...helpers import batchify, frozendict
from ...model.explanation import Explanation, GraphExplanation
from ...model.graph import Graph, Node
from ...model.similarity import GraphSim
from ...typing import BatchSimFunc, Float
from .._explain import Explainable, explain
from .._nested import NestedMeasure
from ..wrappers import transpose_value
from ._matchers import (
    EDGE_MEASURES,
    GRAPH_MATCHERS,
    AnyElement,
    EdgeSimFunc,
    ElementMatcher,
    SemanticEdge,
)

__all__ = [
    "BaseGraphEditFunc",
    "BaseGraphSimFunc",
    "PairSim",
    "PairSimilarities",
    "extend_maximal",
]

type PairSim[K: Hashable] = Mapping[tuple[K, K], float]


def _mutually_unique_mapping[K: Hashable](
    candidates: Iterable[tuple[K, K]],
) -> frozendict[K, K]:
    """Keeps the candidates whose two sides have no competitor.

    Such a pair is the only choice either of its elements has, so every legal
    mapping that maps them at all maps them onto each other.
    """
    pairs = list(candidates)
    y_counts = Counter(y_key for y_key, _ in pairs)
    x_counts = Counter(x_key for _, x_key in pairs)

    return frozendict(
        pair for pair in pairs if y_counts[pair[0]] == 1 and x_counts[pair[1]] == 1
    )


@dataclass(slots=True, frozen=True)
class PairSimilarities[K: Hashable]:
    """Similarities of all legal node and edge pairs of a graph pair.

    The pairs are keyed as `(query key, case key)`.

    Args:
        nodes: Similarities of the node pairs as floats.
        edges: Similarities of the edge pairs.
        node_values: Similarities of the node pairs as returned by the node measure.
    """

    nodes: PairSim[K]
    edges: PairSim[K]
    node_values: Mapping[tuple[K, K], Float]


def _induced_edge_candidates[K: Hashable, N, E, G](
    case: Graph[K, N, E, G],
    query: Graph[K, N, E, G],
    node_mapping: Mapping[K, K],
    edge_matcher: ElementMatcher[E],
) -> list[tuple[K, K]]:
    return [
        (y_value.key, x_value.key)
        for y_value, x_value in itertools.product(
            query.edges.values(), case.edges.values()
        )
        if edge_matcher(x_value.value, y_value.value)
        and x_value.source.key == node_mapping.get(y_value.source.key)
        and x_value.target.key == node_mapping.get(y_value.target.key)
    ]


def _induced_edge_mapping[K: Hashable, N, E, G](
    case: Graph[K, N, E, G],
    query: Graph[K, N, E, G],
    node_mapping: Mapping[K, K],
    edge_matcher: ElementMatcher[E],
    edge_pair_sims: PairSim[K],
) -> frozendict[K, K]:
    candidates = _induced_edge_candidates(case, query, node_mapping, edge_matcher)
    unique = _mutually_unique_mapping(candidates)

    # Without parallel edges nothing competes, so the mapping is already injective
    # as required by Bergmann and Gil (2014) and no choice has to be made.
    if len(unique) == len(candidates):
        return unique

    return frozendict(_best_edge_matching(candidates, edge_pair_sims))


def extend_maximal[K: Hashable](
    mapping: dict[K, K],
    candidates: Iterable[tuple[K, K]],
) -> dict[K, K]:
    """Claims in place every candidate whose two sides are both still free.

    All measures prefer the larger of two equally good mappings, so a pair that was
    left out although nothing competes for it is added here.
    """
    mapped_x = set(mapping.values())

    for y_key, x_key in candidates:
        if y_key not in mapping and x_key not in mapped_x:
            mapping[y_key] = x_key
            mapped_x.add(x_key)

    return mapping


def _best_edge_matching[K: Hashable](
    candidates: Sequence[tuple[K, K]],
    edge_pair_sims: PairSim[K],
) -> dict[K, K]:
    """Pick the most similar injective subset of competing edge candidates.

    Parallel edges make several query edges consistent with the same case edge.
    Pairing them off is a linear assignment problem in which a forbidden combination
    scores zero, exactly like an unmapped edge does in the global similarity, so that
    a valuable pair is never given up for two worthless ones.
    """
    y_keys = list(dict.fromkeys(y_key for y_key, _ in candidates))
    x_keys = list(dict.fromkeys(x_key for _, x_key in candidates))
    y_index = {key: i for i, key in enumerate(y_keys)}
    x_index = {key: i for i, key in enumerate(x_keys)}

    weights = np.zeros((len(y_keys), len(x_keys)))

    for pair in candidates:
        weights[y_index[pair[0]], x_index[pair[1]]] = edge_pair_sims.get(pair, 0.0)

    rows, cols = scipy.optimize.linear_sum_assignment(weights, maximize=True)
    allowed = set(candidates)

    # Candidates worth nothing are interchangeable with a forbidden combination, so
    # the assignment may leave them out and `extend_maximal` picks them back up.
    return extend_maximal(
        {
            y_keys[row]: x_keys[col]
            for row, col in zip(rows.tolist(), cols.tolist(), strict=True)
            if (y_keys[row], x_keys[col]) in allowed
        },
        candidates,
    )


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseGraphEditFunc[K: Hashable, N, E, G]:
    """Base class providing graph edit cost parameters.

    Args:
        node_deletion_cost: Cost of deleting a node of the query.
        node_insertion_cost: Cost of inserting a node of the case.
        edge_deletion_cost: Cost of deleting an edge of the query.
        edge_insertion_cost: Cost of inserting an edge of the case.
    """

    node_deletion_cost: float = 1.0
    node_insertion_cost: float = 0.0
    edge_deletion_cost: float = 1.0
    edge_insertion_cost: float = 0.0

    @property
    def node_weight(self) -> float:
        """Weight of a node pair, the edit costs it saves when fully similar."""
        return self.node_deletion_cost + self.node_insertion_cost

    @property
    def edge_weight(self) -> float:
        """Weight of an edge pair, the edit costs it saves when fully similar."""
        return self.edge_deletion_cost + self.edge_insertion_cost

    def cost_upper_bound(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
    ) -> float:
        """Computes the upper bound on edit cost assuming all elements are deleted or inserted."""
        return (
            len(query.nodes) * self.node_deletion_cost
            + len(case.nodes) * self.node_insertion_cost
            + len(query.edges) * self.edge_deletion_cost
            + len(case.edges) * self.edge_insertion_cost
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseGraphSimFunc[K: Hashable, N, E, G](
    ABC,
    BaseGraphEditFunc[K, N, E, G],
    Explainable[Graph[K, N, E, G]],
    BatchSimFunc[Graph[K, N, E, G], GraphSim[K]],
):
    """Base class for graph similarity functions with node and edge matching.

    Every graph measure is a batch function: it computes the node and edge
    similarities of all graph pairs at once, so batch measures such as embeddings
    work across a whole case base, and then maps every pair on its own with `solve`.

    Args:
        node_measure: Similarity of the values of two nodes.
        edge_measure: Similarity of two edges, by default the mean similarity of their endpoints.
        node_matcher: Whether two nodes may be mapped onto each other, any pair by default.
        edge_matcher: Whether two edges may be mapped onto each other, any pair by default.
    """

    node_measure: NestedMeasure[N, Float]
    edge_measure: Annotated[EdgeSimFunc[E], EDGE_MEASURES] = field(
        default_factory=SemanticEdge
    )
    node_matcher: Annotated[ElementMatcher[N], GRAPH_MATCHERS] = field(
        default_factory=AnyElement
    )
    edge_matcher: Annotated[ElementMatcher[E], GRAPH_MATCHERS] = field(
        default_factory=AnyElement
    )
    _batch_node_measure: BatchSimFunc[Node[K, N], Float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_batch_node_measure",
            batchify(transpose_value(self.node_measure)),
        )

    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]], /
    ) -> list[GraphSim[K]]:
        return [
            self.solve(x, y, sims)
            for (x, y), sims in zip(
                batches, self.batch_pair_similarities(batches), strict=True
            )
        ]

    @abstractmethod
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        """Maps a single graph pair given the similarities of its legal pairs."""
        ...

    @override
    def explain(
        self, case: Graph[K, N, E, G], query: Graph[K, N, E, G], similarity: Float
    ) -> Explanation:
        """Breaks a graph similarity down by the mapping of the query elements, the nodes recursively."""
        if not isinstance(similarity, GraphSim):
            return Explanation(similarity=float(similarity))

        sim: GraphSim[K] = similarity
        graph = GraphExplanation(
            exact=sim.exact,
            nodes=[
                explain(
                    case.nodes[sim.mapping[key]].value,
                    query.nodes[key].value,
                    sim.similarities[key],
                    self.node_measure,
                ).model_copy(update={"query": key, "case": sim.mapping[key]})
                for key in query.nodes
                if key in sim.mapping
            ],
            edges=[
                Explanation(
                    similarity=sim.edge_similarities[key],
                    query=key,
                    case=sim.edge_mapping[key],
                )
                for key in query.edges
                if key in sim.edge_mapping
            ],
            unmapped_nodes=[key for key in query.nodes if key not in sim.mapping],
            unmapped_edges=[key for key in query.edges if key not in sim.edge_mapping],
            node_coverage=len(sim.mapping) / len(query.nodes) if query.nodes else 1.0,
            edge_coverage=(
                len(sim.edge_mapping) / len(query.edges) if query.edges else 1.0
            ),
        )

        return Explanation(similarity=float(sim), graph=graph)

    def induced_edge_mapping(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_mapping: Mapping[K, K],
        edge_pair_sims: PairSim[K],
    ) -> frozendict[K, K]:
        """Derives an injective edge mapping induced by the given node mapping.

        Parallel edges let several candidates compete, which is resolved in favor of
        the most similar combination.
        """
        return _induced_edge_mapping(
            case, query, node_mapping, self.edge_matcher, edge_pair_sims
        )

    def node_pairs(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
    ) -> list[tuple[K, K]]:
        """Returns all legal node pairs between the two graphs."""
        return [
            (y_key, x_key)
            for x_key, y_key in itertools.product(case.nodes.keys(), query.nodes.keys())
            if self.node_matcher(case.nodes[x_key].value, query.nodes[y_key].value)
        ]

    def edge_pairs(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_pairs: Collection[tuple[K, K]],
    ) -> list[tuple[K, K]]:
        """Returns all legal edge pairs consistent with the given node pairs."""
        return [
            (y_key, x_key)
            for x_key, y_key in itertools.product(case.edges.keys(), query.edges.keys())
            if self.edge_matcher(case.edges[x_key].value, query.edges[y_key].value)
            and (query.edges[y_key].source.key, case.edges[x_key].source.key)
            in node_pairs
            and (query.edges[y_key].target.key, case.edges[x_key].target.key)
            in node_pairs
        ]

    def batch_pair_similarities(
        self,
        batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]],
    ) -> list[PairSimilarities[K]]:
        """Computes the legal pair similarities of many graph pairs at once.

        The node similarity function is called once for all node pairs of all
        graph pairs, which lets batch functions such as embeddings work across a
        whole case base, and the same holds for the edge similarity function.
        """
        node_pairs = [self.node_pairs(x, y) for x, y in batches]
        node_values = self._batch_node_measure(
            [
                (x.nodes[x_key], y.nodes[y_key])
                for (x, y), pairs in zip(batches, node_pairs, strict=True)
                for y_key, x_key in pairs
            ]
        )
        node_value_iter = iter(node_values)
        node_value_maps = [
            {pair: next(node_value_iter) for pair in pairs} for pairs in node_pairs
        ]
        node_sim_maps = [
            {pair: float(value) for pair, value in values.items()}
            for values in node_value_maps
        ]
        edge_pairs = [
            self.edge_pairs(x, y, node_sims.keys())
            for (x, y), node_sims in zip(batches, node_sim_maps, strict=True)
        ]
        edge_sims = iter(
            self.edge_measure(
                [
                    (
                        x.edges[x_key].value,
                        y.edges[y_key].value,
                        node_sims[
                            (y.edges[y_key].source.key, x.edges[x_key].source.key)
                        ],
                        node_sims[
                            (y.edges[y_key].target.key, x.edges[x_key].target.key)
                        ],
                    )
                    for (x, y), node_sims, pairs in zip(
                        batches, node_sim_maps, edge_pairs, strict=True
                    )
                    for y_key, x_key in pairs
                ]
            )
        )

        return [
            PairSimilarities(
                node_sims,
                {pair: next(edge_sims) for pair in pairs},
                node_values,
            )
            for node_sims, node_values, pairs in zip(
                node_sim_maps, node_value_maps, edge_pairs, strict=True
            )
        ]

    def similarity(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_mapping: frozendict[K, K],
        sims: PairSimilarities[K],
        *,
        edge_mapping: frozendict[K, K] | None = None,
        exact: bool = False,
    ) -> GraphSim[K]:
        """Builds the result of a mapping, keeping the node similarities as computed.

        Without an edge mapping, the edges are those induced by the node mapping, see `induced_edge_mapping`.
        """
        if edge_mapping is None:
            edge_mapping = self.induced_edge_mapping(
                case, query, node_mapping, sims.edges
            )

        return GraphSim(
            value=self.similarity_value(
                case, query, node_mapping, edge_mapping, sims.nodes, sims.edges
            ),
            similarities=frozendict(
                (y_key, sims.node_values[(y_key, x_key)])
                for y_key, x_key in node_mapping.items()
            ),
            mapping=node_mapping,
            edge_similarities=frozendict(
                (y_key, sims.edges[(y_key, x_key)])
                for y_key, x_key in edge_mapping.items()
            ),
            edge_mapping=edge_mapping,
            exact=exact,
        )

    def similarity_value(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        node_mapping: Mapping[K, K],
        edge_mapping: Mapping[K, K],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> float:
        """Aggregates the mapped similarities without building a `GraphSim`.

        Mapping a pair saves the deletion and insertion costs of its two elements,
        scaled by their similarity, so the costs act as element weights and the
        value is one minus the normalized edit distance, which stays within [0, 1].

        The search based measures need nothing but this value for every state they
        generate, which is why it is kept free of the per-element bookkeeping.
        """
        upper_bound = self.cost_upper_bound(case, query)

        # An upper bound of zero means there is nothing the query asks for,
        # which is trivially satisfied by any case.
        if upper_bound <= 0:
            return 1.0

        return (
            self.node_weight
            * math.fsum(node_pair_sims[pair] for pair in node_mapping.items())
            + self.edge_weight
            * math.fsum(edge_pair_sims[pair] for pair in edge_mapping.items())
        ) / upper_bound
