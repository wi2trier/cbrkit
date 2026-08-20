from collections.abc import Sequence
from dataclasses import dataclass

from frozendict import frozendict

from ...model.graph import Graph, Node, to_sequence
from ...typing import SimFunc
from ..collections import SequenceSim
from ..collections import dtw as collections_dtw
from ..collections import smith_waterman as collections_smith_waterman
from ..collections import twed as collections_twed
from .common import BaseGraphSimFunc, GraphSim, PairSim, extend_maximal

__all__ = [
    "dtw",
    "smith_waterman",
    "twed",
]

type NodeAlignment[K, N] = Sequence[tuple[Node[K, N] | None, Node[K, N] | None]]


@dataclass(slots=True)
class BaseAlignmentSimFunc[K, N, E, G](
    BaseGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
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

    def align(
        self,
        sequence_x: Sequence[Node[K, N]],
        sequence_y: Sequence[Node[K, N]],
        node_pair_sims: PairSim[K],
    ) -> SequenceSim[Node[K, N], float]:
        """Align the two node sequences, returning the alignment as a mapping."""
        raise NotImplementedError

    def legal_node_mapping(
        self,
        alignment: NodeAlignment[K, N],
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

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        sequence_x, _ = to_sequence(x)
        sequence_y, _ = to_sequence(y)
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        alignment = self.align(sequence_x, sequence_y, node_pair_sims).mapping or []
        node_mapping = self.legal_node_mapping(alignment, node_pair_sims)

        return self.similarity(
            x,
            y,
            node_mapping,
            self.induced_edge_mapping(x, y, node_mapping, edge_pair_sims),
            node_pair_sims,
            edge_pair_sims,
        )


@dataclass(slots=True)
class dtw[K, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on Dynamic Time Warping of the node sequences.

    The node similarity is converted into a distance, since DTW minimizes a cost.
    DTW may warp several query nodes onto the same case node, in which case only the
    most similar of these pairs is kept to obtain a legal mapping.

    Args:
        node_sim_func: A similarity function for node values.
        edge_sim_func: A similarity function for edges.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
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
        >>> sim = dtw(node_sim_func)
        >>> sim(graph_x, graph_x)
        GraphSim(value=1.0,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 1.0}),
            edge_similarities=frozendict.frozendict({'e1': 1.0}))
        >>> sim(graph_x, graph_y)
        GraphSim(value=0.5,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 0.0}),
            edge_similarities=frozendict.frozendict({'e1': 0.5}))
    """

    def align(
        self,
        sequence_x: Sequence[Node[K, N]],
        sequence_y: Sequence[Node[K, N]],
        node_pair_sims: PairSim[K],
    ) -> SequenceSim[Node[K, N], float]:
        def distance_func(x_node: Node[K, N], y_node: Node[K, N]) -> float:
            return 1.0 - node_pair_sims.get((y_node.key, x_node.key), 0.0)

        return collections_dtw(distance_func)(
            sequence_x, sequence_y, return_alignment=True
        )


@dataclass(slots=True)
class twed[K, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on the Time Warp Edit Distance of the node sequences.

    In contrast to `dtw`, nodes without a good counterpart are deleted instead of being
    warped onto an already mapped node, so the alignment is injective by construction.
    The node similarity is converted into a distance, since TWED minimizes a cost.

    Args:
        node_sim_func: A similarity function for node values.
        edge_sim_func: A similarity function for edges.
        stiffness: Elasticity along the node sequence, called nu in the TWED paper.
        penalty: Constant cost of deleting a node, called lambda in the TWED paper.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
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
        >>> sim = twed(node_sim_func)
        >>> sim(graph_x, graph_x)
        GraphSim(value=1.0,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 1.0}),
            edge_similarities=frozendict.frozendict({'e1': 1.0}))
        >>> sim(graph_x, graph_y)
        GraphSim(value=0.5,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 0.0}),
            edge_similarities=frozendict.frozendict({'e1': 0.5}))
    """

    stiffness: float = 0.001
    penalty: float = 1.0

    def align(
        self,
        sequence_x: Sequence[Node[K, N]],
        sequence_y: Sequence[Node[K, N]],
        node_pair_sims: PairSim[K],
    ) -> SequenceSim[Node[K, N], float]:
        def distance_func(x_node: Node[K, N], y_node: Node[K, N]) -> float:
            return 1.0 - node_pair_sims.get((y_node.key, x_node.key), 0.0)

        return collections_twed(
            distance_func, stiffness=self.stiffness, penalty=self.penalty
        )(sequence_x, sequence_y, return_alignment=True)


@dataclass(slots=True)
class smith_waterman[K, N, E, G](BaseAlignmentSimFunc[K, N, E, G]):
    """
    Graph similarity function based on the Smith-Waterman alignment of the node sequences.

    The local alignment identifies the best matching sub-sequences, so nodes outside
    the aligned region remain unmapped and contribute a similarity of zero.

    Args:
        node_sim_func: A similarity function for node values.
        edge_sim_func: A similarity function for edges.
        deletion_penalty: Score of skipping a node of the query.
        insertion_penalty: Score of skipping a node of the case.

    Examples:
        >>> from ...model.graph import from_dict
        >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
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
        >>> sim = smith_waterman(node_sim_func)
        >>> sim(graph_x, graph_x)
        GraphSim(value=1.0,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 1.0}),
            edge_similarities=frozendict.frozendict({'e1': 1.0}))
        >>> sim(graph_x, graph_y)
        GraphSim(value=0.5,
            node_mapping=frozendict.frozendict({'1': '1', '2': '2'}),
            edge_mapping=frozendict.frozendict({'e1': 'e1'}),
            node_similarities=frozendict.frozendict({'1': 1.0, '2': 0.0}),
            edge_similarities=frozendict.frozendict({'e1': 0.5}))
    """

    deletion_penalty: float = -1.0
    insertion_penalty: float = -1.0

    def align(
        self,
        sequence_x: Sequence[Node[K, N]],
        sequence_y: Sequence[Node[K, N]],
        node_pair_sims: PairSim[K],
    ) -> SequenceSim[Node[K, N], float]:
        def element_similarity(x_node: Node[K, N], y_node: Node[K, N]) -> float:
            return node_pair_sims.get((y_node.key, x_node.key), 0.0)

        return collections_smith_waterman(
            element_similarity,
            deletion_penalty=self.deletion_penalty,
            insertion_penalty=self.insertion_penalty,
        )(sequence_x, sequence_y, return_alignment=True)
