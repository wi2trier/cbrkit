from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

from pydantic import Field

from ...helpers import frozendict
from ...model.graph import Graph, to_sequence
from ...model.similarity import Alignment, GraphSim, SequenceSim
from ...schema import serializable
from ...typing import SimFunc
from ..collections import DTW, TWED, SmithWaterman
from ._base import BaseGraphSimFunc, PairSim, PairSimilarities, extend_maximal

__all__ = [
    "GraphDTW",
    "GraphSmithWaterman",
    "GraphTWED",
]


@dataclass(slots=True, frozen=True)
class AlignedNode[K: Hashable]:
    """A node of a graph pair as an element of the aligned sequences.

    Args:
        key: Key of the node.
        sims: Similarities of the legal node pairs of the graph pair.
    """

    key: K
    sims: PairSim[K] = field(repr=False, compare=False)


def _node_similarity[K: Hashable](case: AlignedNode[K], query: AlignedNode[K]) -> float:
    """Looks up the similarity of a case node and a query node, zero for an illegal pair."""
    return case.sims.get((query.key, case.key), 0.0)


type Aligner[K: Hashable] = SimFunc[
    Sequence[AlignedNode[K]], SequenceSim[AlignedNode[K], float]
]
"""A sequence measure that aligns the nodes of a graph pair."""


@dataclass(slots=True, frozen=True, kw_only=True)
class BaseAlignmentSimFunc[K: Hashable, N, E, G](BaseGraphSimFunc[K, N, E, G], ABC):
    """
    Base class for graph similarity functions that align the graphs as node sequences.

    Both graphs are converted to sequences via `cbrkit.model.graph.to_sequence`, which
    requires them to be sequential.
    The resulting node alignment is turned into a legal mapping as defined by
    [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005), i.e. a partial
    and injective mapping whose edges are induced by the mapped nodes, and aggregated
    by the shared `similarity` method so that the result is comparable to the other
    graph similarity functions.
    """

    _aligner: Aligner[K] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        super(BaseAlignmentSimFunc, self).__post_init__()
        object.__setattr__(self, "_aligner", self.aligner(_node_similarity))

    @abstractmethod
    def aligner(self, element_measure: SimFunc[AlignedNode[K], float]) -> Aligner[K]:
        """The sequence measure that aligns the nodes by the given node similarity."""
        ...

    def legal_node_mapping(
        self,
        alignment: Alignment[AlignedNode[K]],
        node_pair_sims: PairSim[K],
    ) -> frozendict[K, K]:
        """
        Turn an alignment into a legal node mapping.

        Pairs rejected by the node matcher are dropped, and since warping alignments
        may map several query nodes onto the same case node, the pairs are claimed in
        decreasing order of similarity to keep the mapping injective.
        """
        candidates = sorted(
            (
                (node_pair_sims[(y_node.key, x_node.key)], y_node.key, x_node.key)
                for y_node, x_node in alignment
                if y_node is not None
                and x_node is not None
                and (y_node.key, x_node.key) in node_pair_sims
            ),
            key=lambda candidate: candidate[0],
            reverse=True,
        )

        return frozendict(
            extend_maximal({}, ((y_key, x_key) for _, y_key, x_key in candidates))
        )

    @override
    def solve(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
    ) -> GraphSim[K]:
        sequence_x, _ = to_sequence(case)
        sequence_y, _ = to_sequence(query)
        alignment = (
            self._aligner(
                [AlignedNode(node.key, sims.nodes) for node in sequence_x],
                [AlignedNode(node.key, sims.nodes) for node in sequence_y],
            ).mapping
            or ()
        )
        node_mapping = self.legal_node_mapping(alignment, sims.nodes)

        return self.similarity(case, query, node_mapping, sims)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class GraphDTW[K: Hashable, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on Dynamic Time Warping of the node sequences.

    The node similarity is converted into a distance, since DTW minimizes a cost.
    DTW may warp several query nodes onto the same case node, in which case only the
    most similar of these pairs is kept to obtain a legal mapping.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_measure = lambda n1, n2: 1.0 if n1 == n2 else 0.0
        >>> data_x = {
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> data_y = {
        ...     "nodes": {"1": "A", "2": "C"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> graph_x = from_dict(data_x)
        >>> graph_y = from_dict(data_y)
        >>> sim = GraphDTW(node_measure=node_measure)
        >>> sim([(graph_x, graph_x)])[0]
        GraphSim(value=1.0,
            similarities=frozendict({'1': 1.0, '2': 1.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 1.0}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
        >>> sim([(graph_x, graph_y)])[0]
        GraphSim(value=0.5,
            similarities=frozendict({'1': 1.0, '2': 0.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 0.5}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
    """

    kind: Literal["graph_dtw"] = field(default="graph_dtw", repr=False)

    @override
    def aligner(self, element_measure: SimFunc[AlignedNode[K], float]) -> Aligner[K]:
        return DTW(element_measure=element_measure, alignment=True)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class GraphTWED[K: Hashable, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on the Time Warp Edit Distance of the node sequences.

    In contrast to `GraphDTW`, nodes without a good counterpart are deleted instead of being
    warped onto an already mapped node, so the alignment is injective by construction.
    The node similarity is converted into a distance, since TWED minimizes a cost.

    Args:
        stiffness: Elasticity along the node sequence, called nu in the TWED paper.
        penalty: Constant cost of deleting a node, called lambda in the TWED paper.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_measure = lambda n1, n2: 1.0 if n1 == n2 else 0.0
        >>> data_x = {
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> data_y = {
        ...     "nodes": {"1": "A", "2": "C"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> graph_x = from_dict(data_x)
        >>> graph_y = from_dict(data_y)
        >>> sim = GraphTWED(node_measure=node_measure)
        >>> sim([(graph_x, graph_x)])[0]
        GraphSim(value=1.0,
            similarities=frozendict({'1': 1.0, '2': 1.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 1.0}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
        >>> sim([(graph_x, graph_y)])[0]
        GraphSim(value=0.5,
            similarities=frozendict({'1': 1.0, '2': 0.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 0.5}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
    """

    stiffness: Annotated[float, Field(ge=0)] = 0.001
    penalty: Annotated[float, Field(ge=0)] = 1.0
    kind: Literal["graph_twed"] = field(default="graph_twed", repr=False)

    @override
    def aligner(self, element_measure: SimFunc[AlignedNode[K], float]) -> Aligner[K]:
        return TWED(
            element_measure=element_measure,
            stiffness=self.stiffness,
            penalty=self.penalty,
            alignment=True,
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class GraphSmithWaterman[K: Hashable, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on the Smith-Waterman alignment of the node sequences.

    The local alignment identifies the best matching sub-sequences, so nodes outside
    the aligned region remain unmapped and contribute a similarity of zero.

    Args:
        deletion_penalty: Score of skipping a node of the query.
        insertion_penalty: Score of skipping a node of the case.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_measure = lambda n1, n2: 1.0 if n1 == n2 else 0.0
        >>> data_x = {
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> data_y = {
        ...     "nodes": {"1": "A", "2": "C"},
        ...     "edges": {"e1": {"source": "1", "target": "2", "value": None}},
        ...     "value": None
        ... }
        >>> graph_x = from_dict(data_x)
        >>> graph_y = from_dict(data_y)
        >>> sim = GraphSmithWaterman(node_measure=node_measure)
        >>> sim([(graph_x, graph_x)])[0]
        GraphSim(value=1.0,
            similarities=frozendict({'1': 1.0, '2': 1.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 1.0}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
        >>> sim([(graph_x, graph_y)])[0]
        GraphSim(value=0.5,
            similarities=frozendict({'1': 1.0, '2': 0.0}),
            mapping=frozendict({'1': '1', '2': '2'}),
            edge_similarities=frozendict({'e1': 0.5}),
            edge_mapping=frozendict({'e1': 'e1'}),
            exact=False)
    """

    deletion_penalty: float = -1.0
    insertion_penalty: float = -1.0
    kind: Literal["graph_smith_waterman"] = field(
        default="graph_smith_waterman", repr=False
    )

    @override
    def aligner(self, element_measure: SimFunc[AlignedNode[K], float]) -> Aligner[K]:
        return SmithWaterman(
            element_measure=element_measure,
            deletion_penalty=self.deletion_penalty,
            insertion_penalty=self.insertion_penalty,
            alignment=True,
        )
