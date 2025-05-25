from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

from frozendict import frozendict

from ...helpers import (
    batchify_sim,
    optional_dependencies,
    unbatchify_sim,
    unpack_float,
    unpack_floats,
)
from ...model.graph import Edge, Graph, Node, to_sequence
from ...typing import AnySimFunc, BatchSimFunc, Float, SimFunc
from ..wrappers import transpose_value
from .common import GraphSim

__all__ = [
    "dtw",
    "smith_waterman",
]


@dataclass(slots=True, frozen=True)
class SimilarityData[K]:
    """Dataclass to hold similarity data."""

    node_mapping: dict[K, K]
    node_similarities: dict[K, float]
    edge_mapping: dict[K, K]
    edge_similarities: dict[K, float]
    node_similarity: float
    edge_similarity: float | None


def _build_node_edge_mappings_and_similarity[K, N, E](
    alignment: Sequence[tuple[Node[K, N], Node[K, N]]] | None,
    node_sim_func: BatchSimFunc[Node[K, N], Float],
    edges_x: list[Edge[K, N, E]],
    edges_y: list[Edge[K, N, E]],
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float] | None = None,
) -> SimilarityData:
    """
    Internal helper for building node/edge mappings & computing average similarities.

    alignment: a list of (xn, yn) pairs from some alignment. Some pairs may be (None, y) or (x, None).
    node_sim_func: can be batch or single, we'll batchify it here.
    edges_x, edges_y: the edges from each graph's sequence, typically parallel in simple alignment.
    edge_sim_func: optional, can be batch or single.

    Returns:
        SimilarityData object containing node/edge mappings and similarities.
    """
    node_pairs = [(xn, yn) for (xn, yn) in alignment if xn and yn] if alignment else []
    node_sims = node_sim_func(node_pairs) if node_pairs else []
    node_similarity = (
        sum(unpack_floats(node_sims)) / len(node_sims) if node_sims else 0.0
    )

    node_mapping: dict = {}
    node_similarities: dict = {}

    for (xn, yn), sim_val in zip(node_pairs, node_sims, strict=True):
        node_mapping[yn.key] = xn.key
        node_similarities[yn.key] = unpack_float(sim_val)

    edge_similarity = None
    edge_mapping: dict = {}
    edge_similarities: dict = {}

    if edge_sim_func:
        edge_pairs = list(zip(edges_x, edges_y, strict=True))
        edge_sims = edge_sim_func(edge_pairs) if edge_pairs else []
        edge_similarity = (
            sum(unpack_floats(edge_sims)) / len(edge_sims) if edge_sims else 0.0
        )

        for (xe, ye), es in zip(edge_pairs, edge_sims, strict=True):
            if xe and ye:
                edge_mapping[ye.key] = xe.key
                edge_similarities[ye.key] = unpack_float(es)

    return SimilarityData(
        node_mapping,
        node_similarities,
        edge_mapping,
        edge_similarities,
        node_similarity,
        edge_similarity,
    )


with optional_dependencies():
    from ..collections import dtw as collections_dtw

    @dataclass(slots=True, frozen=True)
    class dtw[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
        """
        Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

        Args:
            node_sim_func: A similarity function for nodes (SimFunc or BatchSimFunc).
            edge_sim_func: An optional similarity function for edges (SimFunc or BatchSimFunc).
            normalize: Whether to normalize the similarity score by the alignment length (default: True).

        Examples:
            >>> from ...model.graph import Node, Edge, Graph, SerializedGraph, from_dict
            >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
            >>> edge_sim_func = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0

            >>> data_x = {
            ...     "nodes": {
            ...         "1": "A",
            ...         "2": "B",
            ...     },
            ...     "edges": {
            ...         "e1": {"source": "1", "target": "2", "value": "X"},
            ...     },
            ...     "value": None
            ... }
            >>> data_y = {
            ...     "nodes": {
            ...         "1": "A",
            ...         "2": "C",
            ...     },
            ...     "edges": {
            ...         "e1": {"source": "1", "target": "2", "value": "Y"},
            ...     },
            ...     "value": None
            ... }
            >>> graph_x = from_dict(data_x)
            >>> graph_y = from_dict(data_y)

            >>> g_dtw = dtw(node_sim_func, edge_sim_func)
            >>> result = g_dtw(graph_x, graph_y)
            >>> print(result)
            GraphSim(value=0.05555555555555555,
                node_mapping=frozendict.frozendict({'1': '2', '2': '2'}), edge_mapping=frozendict.frozendict({'e1': 'e1'}),
                node_similarities=frozendict.frozendict({'1': 0.0, '2': 0.0}), edge_similarities=frozendict.frozendict({'e1': 0.0}))
        """

        node_sim_func: AnySimFunc[N, Float]
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None
        normalize: bool = True

        def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
            sequence_x, edges_x = to_sequence(x)
            sequence_y, edges_y = to_sequence(y)
            node_sim_func: BatchSimFunc[Node[K, N], Float] = batchify_sim(
                transpose_value(self.node_sim_func)
            )
            edge_sim_func = (
                batchify_sim(self.edge_sim_func) if self.edge_sim_func else None
            )

            # 3) Use DTW for nodes, with alignment
            node_result = collections_dtw(node_sim_func)(
                sequence_x, sequence_y, return_alignment=True
            )
            alignment = node_result.mapping  # matched (x_node, y_node)

            # 4) Build node/edge mapping & average similarity using the helper
            similarity_data = _build_node_edge_mappings_and_similarity(
                alignment, node_sim_func, edges_x, edges_y, edge_sim_func
            )

            # 5) Combine node & edge similarity
            total_similarity = (
                (similarity_data.node_similarity + similarity_data.edge_similarity)
                / 2.0
                if similarity_data.edge_similarity is not None
                else similarity_data.node_similarity
            )

            # 6) Normalize by alignment length if applicable
            if self.normalize and alignment:
                alen = len([pair for pair in alignment if pair[0] and pair[1]])
                if alen > 0:
                    total_similarity /= alen

            return GraphSim(
                value=total_similarity,
                node_mapping=frozendict(similarity_data.node_mapping),
                edge_mapping=frozendict(similarity_data.edge_mapping),
                node_similarities=frozendict(similarity_data.node_similarities),
                edge_similarities=frozendict(similarity_data.edge_similarities),
            )


with optional_dependencies():
    from ..collections import smith_waterman as collections_smith_waterman

    @dataclass(slots=True)
    class smith_waterman[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
        """
        Graph-based Smith-Waterman similarity function leveraging local sequence alignment
        plus ProCAKE-style weighting for local similarities.

        Args:
            node_sim_func: Base node similarity function (SimFunc or BatchSimFunc).
            dataflow_in_sim_func: Similarity for incoming data flow (optional).
            dataflow_out_sim_func: Similarity for outgoing data flow (optional).
            l_t: Weight for task/node similarity. Default = 1.0
            l_i: Weight for incoming dataflow similarity. Default = 0.0 (ignored if 0).
            l_o: Weight for outgoing dataflow similarity. Default = 0.0 (ignored if 0).
            edge_sim_func: If provided, an edge similarity function (SimFunc or BatchSimFunc).
            match_score: The SW match score (default 2).
            mismatch_penalty: The SW mismatch penalty (default -1).
            gap_penalty: The SW gap penalty (default -1).
            normalize: Whether to normalize the final similarity (default True).

        Examples:
            >>> from ...model.graph import Node, Edge, Graph, SerializedGraph, from_dict
            >>> node_sim_func = lambda n1, n2: 1.0 if n1 == n2 else 0.0
            >>> edge_sim_func = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0
            >>> def dataflow_dummy(a, b): return 0.5  # pretend data flow sim

            >>> data_x = {
            ...     "nodes": {
            ...         "1": "A",
            ...         "2": "B",
            ...     },
            ...     "edges": {
            ...         "e1": {"source": "1", "target": "2", "value": "X"},
            ...     },
            ...     "value": None
            ... }

            >>> data_y = {
            ...     "nodes": {
            ...         "1": "A",
            ...         "2": "C",
            ...     },
            ...     "edges": {
            ...         "e1": {"source": "1", "target": "2", "value": "X"},
            ...     },
            ...     "value": None
            ... }
            >>> graph_x = from_dict(data_x)
            >>> graph_y = from_dict(data_y)

            >>> g_swa = smith_waterman(
            ...     node_sim_func,
            ...     edge_sim_func,
            ...     dataflow_in_sim_func=dataflow_dummy,
            ...     dataflow_out_sim_func=dataflow_dummy,
            ...     l_t=1.0, l_i=0.5, l_o=0.5,
            ...     use_procake_formula=True
            ... )
            >>> result = g_swa(graph_x, graph_y)
            >>> print(result)
            GraphSim(value=0.015,
                node_mapping=frozendict.frozendict({'1': '1', '2': '2'}), edge_mapping=frozendict.frozendict({'e1': 'e1'}),
                node_similarities=frozendict.frozendict({'1': 0.75, '2': 0.25}), edge_similarities=frozendict.frozendict({'e1': 1.0}))

            >>> # Another example without the ProCAKE weighting:
            >>> g_swa_naive = smith_waterman(
            ...     node_sim_func,
            ...     match_score=2,
            ...     mismatch_penalty=-1,
            ...     gap_penalty=-1,
            ...     normalize=True,
            ...     use_procake_formula=False
            ... )
            >>> result_naive = g_swa_naive(graph_x, graph_y)
            >>> print(result_naive)
            GraphSim(value=0.01,
                node_mapping=frozendict.frozendict({'1': '1', '2': '2'}), edge_mapping=frozendict.frozendict({}),
                node_similarities=frozendict.frozendict({'1': 1.0, '2': 0.0}), edge_similarities=frozendict.frozendict({}))
        """

        node_sim_func: InitVar[AnySimFunc[N, Float]]
        edge_sim_func: InitVar[AnySimFunc[Edge[K, N, E], Float] | None] = None
        batched_node_sim_func: BatchSimFunc[Node[K, N], Float] = field(init=False)
        unbatched_node_sim_func: SimFunc[Node[K, N], Float] = field(init=False)
        batched_edge_sim_func: BatchSimFunc[Edge[K, N, E], Float] | None = field(
            init=False
        )
        dataflow_in_sim_func: SimFunc | None = None
        dataflow_out_sim_func: SimFunc | None = None
        l_t: float = 1.0
        l_i: float = 0.0
        l_o: float = 0.0

        match_score: int = 2
        mismatch_penalty: int = -1
        gap_penalty: int = -1
        normalize: bool = True

        use_procake_formula: bool = False

        def __post_init__(
            self,
            node_sim_func: AnySimFunc[N, Float],
            edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None,
        ) -> None:
            self.batched_node_sim_func = batchify_sim(transpose_value(node_sim_func))
            self.unbatched_node_sim_func = unbatchify_sim(
                transpose_value(node_sim_func)
            )
            self.batched_edge_sim_func = (
                batchify_sim(edge_sim_func) if edge_sim_func else None
            )

        def local_node_sim(self, q_node: Node[K, N], c_node: Node[K, N]) -> float:
            base = unpack_float(self.unbatched_node_sim_func(q_node, c_node))

            if self.use_procake_formula:
                in_flow = 0.0
                if self.l_i != 0.0 and self.dataflow_in_sim_func:
                    # In real usage, pass appropriate node/edge data
                    in_flow = self.dataflow_in_sim_func(None, None)

                out_flow = 0.0
                if self.l_o != 0.0 and self.dataflow_out_sim_func:
                    out_flow = self.dataflow_out_sim_func(None, None)

                total_weight = self.l_t + self.l_i + self.l_o

                if total_weight == 0:
                    return 0.0

                return (
                    self.l_t * base + self.l_i * in_flow + self.l_o * out_flow
                ) / total_weight

            return base

        def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
            # 1) Convert graphs to sequences
            sequence_x, edges_x = to_sequence(x)
            sequence_y, edges_y = to_sequence(y)

            # 3) We'll rely on the minineedle-based SW for a raw score.
            sw_obj = collections_smith_waterman(
                match_score=self.match_score,
                mismatch_penalty=self.mismatch_penalty,
                gap_penalty=self.gap_penalty,
            )

            # Use the length of each sequence as tokens
            tokens_x = list(range(len(sequence_x)))
            tokens_y = list(range(len(sequence_y)))
            raw_sw_score = sw_obj(tokens_x, tokens_y)

            # 4) We do not have a direct alignment from the raw minineedle call,
            #    so let's do a naive 1:1 for the shorter length.
            length = min(len(sequence_x), len(sequence_y))
            alignment = [(sequence_x[i], sequence_y[i]) for i in range(length)]

            # 5) Build node/edge mappings & average similarities using local_node_sim
            similarity_data = _build_node_edge_mappings_and_similarity(
                alignment,
                batchify_sim(self.local_node_sim),
                edges_x,
                edges_y,
                self.batched_edge_sim_func,
            )

            # 6) Combine node & edge sim, scale by raw SW score (if > 0)
            if similarity_data.edge_similarity is not None:
                total_similarity = (
                    similarity_data.node_similarity + similarity_data.edge_similarity
                ) / 2.0
            else:
                total_similarity = similarity_data.node_similarity

            if raw_sw_score > 0:
                total_similarity *= raw_sw_score / 100.0

            # 7) Normalize if requested
            if self.normalize and length > 0:
                total_similarity /= length

            return GraphSim(
                value=total_similarity,
                node_mapping=frozendict(similarity_data.node_mapping),
                edge_mapping=frozendict(similarity_data.edge_mapping),
                node_similarities=frozendict(similarity_data.node_similarities),
                edge_similarities=frozendict(similarity_data.edge_similarities),
            )
