# alignment.py

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from .model import Graph, Node, Edge, GraphSim, to_sequence
from ...helpers import batchify_sim, unbatchify_sim
from ...typing import AnySimFunc, SimFunc, Float
from ..collections import dtw as dtwmodule
from ..collections import smith_waterman as minineedle_sw

__all__ = ["dtw", "smith_waterman"]


@dataclass(slots=True, frozen=True)
class SimilarityData[K]:
    """Dataclass to hold similarity data."""

    node_mapping: Dict[K, K]
    node_similarities: Dict[K, float]
    edge_mapping: Dict[K, K]
    edge_similarities: Dict[K, float]
    node_similarity: float
    edge_similarity: Optional[float]


###############################################################################
# Private helper to build mappings and compute average node/edge similarity.
###############################################################################
def _build_node_edge_mappings_and_similarity[N, E](
    alignment: List[Tuple[Optional[N], Optional[N]]],
    node_sim_func: AnySimFunc,
    edges_x: List[E],
    edges_y: List[E],
    edge_sim_func: Optional[AnySimFunc] = None,
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
    batch_node_sim_func = batchify_sim(node_sim_func)
    node_pairs = [(xn, yn) for (xn, yn) in alignment if xn and yn]
    node_sims = batch_node_sim_func(node_pairs) if node_pairs else []
    node_similarity = sum(node_sims) / len(node_sims) if node_sims else 0.0

    node_mapping: Dict = {}
    node_similarities: Dict = {}
    for (xn, yn), sim_val in zip(node_pairs, node_sims):
        node_mapping[yn.key] = xn.key
        node_similarities[yn.key] = float(sim_val)

    edge_similarity = None
    edge_mapping: Dict = {}
    edge_similarities: Dict = {}
    if edge_sim_func:
        batch_edge_sim_func = batchify_sim(edge_sim_func)
        edge_pairs = list(zip(edges_x, edges_y))
        edge_sims = batch_edge_sim_func(edge_pairs) if edge_pairs else []
        edge_similarity = sum(edge_sims) / len(edge_sims) if edge_sims else 0.0

        for (xe, ye), es in zip(edge_pairs, edge_sims):
            if xe and ye:
                edge_mapping[ye.key] = xe.key
                edge_similarities[ye.key] = float(es)

    return SimilarityData(
        node_mapping,
        node_similarities,
        edge_mapping,
        edge_similarities,
        node_similarity,
        edge_similarity,
    )


@dataclass(slots=True, frozen=True)
class dtw[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

    Args:
        node_sim_func: A similarity function for nodes (SimFunc or BatchSimFunc).
        edge_sim_func: An optional similarity function for edges (SimFunc or BatchSimFunc).
        normalize: Whether to normalize the similarity score by the alignment length (default: True).

    Examples:
        >>> from .model import Node, Edge, Graph, SerializedGraph
        >>> node_similarity = lambda n1, n2: 1.0 if n1.value == n2.value else 0.0
        >>> edge_similarity = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0

        >>> data_x = {
        ...     "nodes": {
        ...         "1": {"value": "A"},
        ...         "2": {"value": "B"},
        ...     },
        ...     "edges": {
        ...         "e1": {"source": "1", "target": "2", "value": "X"},
        ...     },
        ...     "value": None
        ... }
        >>> sx = SerializedGraph.model_validate(data_x)
        >>> graph_x = Graph.load(sx)

        >>> data_y = {
        ...     "nodes": {
        ...         "1": {"value": "A"},
        ...         "2": {"value": "C"},
        ...     },
        ...     "edges": {
        ...         "e1": {"source": "1", "target": "2", "value": "Y"},
        ...     },
        ...     "value": None
        ... }
        >>> sy = SerializedGraph.model_validate(data_y)
        >>> graph_y = Graph.load(sy)

        >>> g_dtw = dtw(node_similarity, edge_similarity)
        >>> result = g_dtw(graph_x, graph_y)
        >>> print(result)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'},
                 edge_mapping={'e1': 'e1'}, node_similarities=..., edge_similarities=...)

        >>> # Example with BatchSimFunc
        >>> def batch_node_similarity(pairs):
        ...     return [1.0 if n1.value == n2.value else 0.0 for n1, n2 in pairs]
        >>> def batch_edge_similarity(pairs):
        ...     return [1.0 if e1.value == e2.value else 0.0 for e1, e2 in pairs]
        >>> g_dtw_batch = dtw(batch_node_similarity, batch_edge_similarity)
        >>> result_batch = g_dtw_batch(graph_x, graph_y)
        >>> print(result_batch)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'},
                 edge_mapping={'e1': 'e1'}, node_similarities=..., edge_similarities=...)
    """

    node_sim_func: AnySimFunc[Node[K, N], Float]
    edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None
    normalize: bool = True

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        # 1) Convert both graphs to node sequences & edges
        sequence_x, edges_x = to_sequence(x)
        sequence_y, edges_y = to_sequence(y)

        # 2) Convert node_sim_func to a pairwise function
        wrapped_node_sim_func = unbatchify_sim(self.node_sim_func)

        # 3) Use DTW for nodes, with alignment
        node_result = dtwmodule(distance_func=wrapped_node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )
        alignment = node_result.mapping  # matched (x_node, y_node)

        # 4) Build node/edge mapping & average similarity using the helper
        similarity_data = _build_node_edge_mappings_and_similarity(
            alignment, self.node_sim_func, edges_x, edges_y, self.edge_sim_func
        )

        # 5) Combine node & edge similarity
        total_similarity = (
            (similarity_data.node_similarity + similarity_data.edge_similarity) / 2.0
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
            node_mapping=similarity_data.node_mapping,
            edge_mapping=similarity_data.edge_mapping,
            node_similarities=similarity_data.node_similarities,
            edge_similarities=similarity_data.edge_similarities,
        )


@dataclass(slots=True, frozen=True)
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
        >>> from .model import Node, Edge, Graph, SerializedGraph
        >>> node_similarity = lambda n1, n2: 1.0 if n1.value == n2.value else 0.0
        >>> edge_sim_func = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0
        >>> def dataflow_dummy(a, b): return 0.5  # pretend data flow sim

        >>> data_x = {
        ...     "nodes": {
        ...         "1": {"value": "A"},
        ...         "2": {"value": "B"},
        ...     },
        ...     "edges": {
        ...         "e1": {"source": "1", "target": "2", "value": "X"},
        ...     },
        ...     "value": None
        ... }
        >>> graph_x = Graph.load(SerializedGraph.model_validate(data_x))

        >>> data_y = {
        ...     "nodes": {
        ...         "1": {"value": "A"},
        ...         "2": {"value": "C"},
        ...     },
        ...     "edges": {
        ...         "e1": {"source": "1", "target": "2", "value": "X"},
        ...     },
        ...     "value": None
        ... }
        >>> graph_y = Graph.load(SerializedGraph.model_validate(data_y))

        >>> g_swa = smith_waterman(
        ...     node_sim_func=node_similarity,
        ...     edge_sim_func=edge_sim_func,
        ...     dataflow_in_sim_func=dataflow_dummy,
        ...     dataflow_out_sim_func=dataflow_dummy,
        ...     l_t=1.0, l_i=0.5, l_o=0.5,
        ...     use_procake_formula=True
        ... )
        >>> result = g_swa(graph_x, graph_y)
        >>> print(result)
        GraphSim(value=0.015, node_mapping={'1': '1', '2': '2'}, edge_mapping={'e1': 'e1'},
                 node_similarities={'1': 0.75, '2': 0.25}, edge_similarities={'e1': 1.0})

        >>> # Another example without the ProCAKE weighting:
        >>> g_swa_naive = smith_waterman(
        ...     node_sim_func=node_similarity,
        ...     match_score=2,
        ...     mismatch_penalty=-1,
        ...     gap_penalty=-1,
        ...     normalize=True,
        ...     use_procake_formula=False
        ... )
        >>> result_naive = g_swa_naive(graph_x, graph_y)
        >>> print(result_naive)
        GraphSim(value=0.01, node_mapping={'1': '1', '2': '2'}, edge_mapping={},
                 node_similarities={'1': 1.0, '2': 0.0}, edge_similarities={})
    """

    node_sim_func: AnySimFunc[Node[K, N], Float]
    edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None
    dataflow_in_sim_func: Optional[AnySimFunc] = None  # No specific typing needed here
    dataflow_out_sim_func: Optional[AnySimFunc] = None  # No specific typing needed here
    l_t: float = 1.0
    l_i: float = 0.0
    l_o: float = 0.0

    match_score: int = 2
    mismatch_penalty: int = -1
    gap_penalty: int = -1
    normalize: bool = True

    use_procake_formula: bool = False

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> GraphSim[K]:
        # 1) Convert graphs to sequences
        sequence_x, edges_x = to_sequence(x)
        sequence_y, edges_y = to_sequence(y)

        # 2) Decide how to compute local similarity for each node pair
        if self.use_procake_formula:

            def local_node_sim(q_node, c_node):
                base = self.node_sim_func(q_node, c_node) if self.node_sim_func else 0.0

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

        else:
            # Naive approach: just call the node_sim_func
            def local_node_sim(q_node, c_node):
                return self.node_sim_func(q_node, c_node) if self.node_sim_func else 0.0

        # 3) We'll rely on the minineedle-based SW for a raw score.
        sw_obj = minineedle_sw(
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
            local_node_sim,
            edges_x,
            edges_y,
            self.edge_sim_func,
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
            node_mapping=similarity_data.node_mapping,
            edge_mapping=similarity_data.edge_mapping,
            node_similarities=similarity_data.node_similarities,
            edge_similarities=similarity_data.edge_similarities,
        )
