from dataclasses import dataclass

from ...helpers import batchify_sim, unbatchify_sim, unpack_floats
from ...typing import AnySimFunc, Float, SimFunc
from ..collections import dtw as dtwmodule
from .model import Edge, Graph, GraphSim, Node, to_sequence

__all__ = ["dtw"]


@dataclass(slots=True, frozen=True)
class dtw[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

    Args:
        node_sim_func: A similarity function for nodes (SimFunc or BatchSimFunc).
        edge_sim_func: An optional similarity function for edges (SimFunc or BatchSimFunc).
        normalize: Whether to normalize the similarity score by the alignment length (default: True).

    Examples:
        >>> from frozendict import frozendict
        >>> from .model import Node, Edge, Graph
        >>> node_similarity = lambda n1, n2: 1.0 if n1.value == n2.value else 0.0
        >>> edge_similarity = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0
        >>> nodes_x = frozendict({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='B'),
        ... })
        >>> edges_x = frozendict({
        ...     'e1': Edge(key='e1', source=nodes_x['1'], target=nodes_x['2'], value='X'),
        ... })
        >>> graph_x = Graph(nodes=nodes_x, edges=edges_x, value=None)
        >>> nodes_y = frozendict({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='C'),
        ... })
        >>> edges_y = frozendict({
        ...     'e1': Edge(key='e1', source=nodes_y['1'], target=nodes_y['2'], value='Y'),
        ... })
        >>> graph_y = Graph(nodes=nodes_y, edges=edges_y, value=None)
        >>> g_dtw = dtw(node_similarity, edge_similarity)
        >>> result = g_dtw(graph_x, graph_y)
        >>> print(result)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'}, edge_mapping={'e1': 'e1'},
                 node_similarities=..., edge_similarities=...)

        >>> # Example with BatchSimFunc
        >>> def batch_node_similarity(pairs):
        ...     return [1.0 if n1.value == n2.value else 0.0 for n1, n2 in pairs]
        >>> def batch_edge_similarity(pairs):
        ...     return [1.0 if e1.value == e2.value else 0.0 for e1, e2 in pairs]
        >>> g_dtw_batch = dtw(batch_node_similarity, batch_edge_similarity)
        >>> result_batch = g_dtw_batch(graph_x, graph_y)
        >>> print(result_batch)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'}, edge_mapping={'e1': 'e1'},
                 node_similarities=..., edge_similarities=...)
    """

    node_sim_func: AnySimFunc[Node[K, N], Float]
    edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None
    normalize: bool = True

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        # Convert both graphs to node sequences and edge lists
        sequence_x, edges_x = to_sequence(x)  # sequence_x: list[N], edges_x: list[E]
        sequence_y, edges_y = to_sequence(y)  # sequence_y: list[N], edges_y: list[E]

        # Convert node_sim_func into a pairwise sim function
        wrapped_node_sim_func = unbatchify_sim(self.node_sim_func)

        # Use DTW for nodes, getting alignment
        node_result = dtwmodule(distance_func=wrapped_node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )
        alignment = node_result.mapping  # matched (x_node, y_node) pairs

        # Batchify for node similarities
        batch_node_sim_func = batchify_sim(self.node_sim_func)
        node_pairs = (
            [(xn, yn) for xn, yn in alignment if xn and yn]
            if alignment is not None
            else []
        )
        node_sims = unpack_floats(batch_node_sim_func(node_pairs))
        node_similarity = sum(node_sims) / len(node_sims) if node_sims else 0.0

        # Build node mapping and node_similarities dict
        node_mapping = {}
        node_similarities = {}
        for (xn, yn), sim in zip(node_pairs, node_sims, strict=True):
            # In the mapping, we store y_node.key -> x_node.key
            node_mapping[yn.key] = xn.key
            # For node_similarities, we likewise index by y_node.key
            node_similarities[yn.key] = float(sim)

        edge_similarity = None
        edge_mapping = {}
        edge_similarities = {}

        if self.edge_sim_func:
            # Convert edge_sim_func into a batched sim function
            batch_edge_sim_func = batchify_sim(self.edge_sim_func)
            edge_pairs = list(zip(edges_x, edges_y, strict=True))

            # Compute edge similarities over all edge pairs
            edge_sims = unpack_floats(batch_edge_sim_func(edge_pairs))
            edge_similarity = sum(edge_sims) / len(edge_sims) if edge_sims else 0.0

            # Build edge mapping and edge_similarities dict
            for (xe, ye), sim in zip(edge_pairs, edge_sims, strict=True):
                if xe and ye:
                    edge_mapping[ye.key] = xe.key
                    edge_similarities[ye.key] = float(sim)

        total_similarity = (
            (node_similarity + edge_similarity) / 2.0
            if edge_similarity is not None
            else node_similarity
        )

        # Apply normalization by the alignment length (if applicable)
        if self.normalize and alignment:
            # Avoid dividing by zero
            alen = len([pair for pair in alignment if pair[0] and pair[1]])
            if alen > 0:
                total_similarity /= alen

        return GraphSim(
            value=total_similarity,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
            node_similarities=node_similarities,
            edge_similarities=edge_similarities,
        )
