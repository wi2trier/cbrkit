from dataclasses import dataclass
from .model import Graph, to_sequence, GraphSim
from ...helpers import dist2sim, batchify_sim, unbatchify_sim
from ...typing import AnySimFunc, BatchSimFunc, SimFunc
from ..collections import dtw as dtwmodule
from collections.abc import Sequence

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
        >>> from immutables import Map
        >>> from .model import Node, Edge, Graph
        >>> node_similarity = lambda n1, n2: 1.0 if n1.value == n2.value else 0.0
        >>> edge_similarity = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0
        >>> nodes_x = Map({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='B'),
        ... })
        >>> edges_x = Map({
        ...     'e1': Edge(key='e1', source=nodes_x['1'], target=nodes_x['2'], value='X'),
        ... })
        >>> graph_x = Graph(nodes=nodes_x, edges=edges_x, value=None)
        >>> nodes_y = Map({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='C'),
        ... })
        >>> edges_y = Map({
        ...     'e1': Edge(key='e1', source=nodes_y['1'], target=nodes_y['2'], value='Y'),
        ... })
        >>> graph_y = Graph(nodes=nodes_y, edges=edges_y, value=None)
        >>> g_dtw = dtw(node_similarity, edge_similarity)
        >>> result = g_dtw(graph_x, graph_y)
        >>> print(result)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'}, edge_mapping={'e1': 'e1'})

        >>> # Example with BatchSimFunc
        >>> def batch_node_similarity(pairs):
        ...     return [1.0 if n1.value == n2.value else 0.0 for n1, n2 in pairs]
        >>> def batch_edge_similarity(pairs):
        ...     return [1.0 if e1.value == e2.value else 0.0 for e1, e2 in pairs]
        >>> g_dtw_batch = dtw(batch_node_similarity, batch_edge_similarity)
        >>> result_batch = g_dtw_batch(graph_x, graph_y)
        >>> print(result_batch)
        GraphSim(value=0.05555555555555555, node_mapping={'1': '1', '2': '2'}, edge_mapping={'e1': 'e1'})
    """

    node_sim_func: AnySimFunc
    edge_sim_func: AnySimFunc | None = None
    normalize: bool = True

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        # Use the updated to_sequence function
        sequence_x, edges_x = to_sequence(x)  # sequence_x: list[N], edges_x: list[E]
        sequence_y, edges_y = to_sequence(y)  # sequence_y: list[N], edges_y: list[E]

        # Convert sim funcs into normal (pairwise) calls
        wrapped_node_sim_func = unbatchify_sim(self.node_sim_func)
        wrapped_edge_sim_func = (
            unbatchify_sim(self.edge_sim_func) if self.edge_sim_func is not None else None
        )

        # Use the dtwmodule for node distances
        node_result = dtwmodule(distance_func=wrapped_node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )

        # Extract alignment and node distance
        alignment = node_result.mapping
        node_distance = node_result.value

        # Batchify for node similarity calculation over alignment pairs
        batch_node_sim_func = batchify_sim(self.node_sim_func)
        node_pairs = [(x_node, y_node) for x_node, y_node in alignment if x_node and y_node]
        node_sims = batch_node_sim_func(node_pairs)
        node_similarity = sum(node_sims) / len(node_sims) if node_sims else 0.0

        node_mapping = {
            y_node.key: x_node.key for x_node, y_node in alignment if x_node and y_node
        }

        edge_similarity = None
        edge_mapping = {}

        if self.edge_sim_func:
            # Use the dtwmodule for edge distances
            edge_result = dtwmodule(distance_func=wrapped_edge_sim_func)(
                edges_x, edges_y, return_alignment=False
            )

            edge_distance = edge_result.value

            # Batchify for edge similarity
            batch_edge_sim_func = batchify_sim(self.edge_sim_func)
            edge_pairs = list(zip(edges_x, edges_y))
            edge_sims = batch_edge_sim_func(edge_pairs)
            edge_similarity = sum(edge_sims) / len(edge_sims) if edge_sims else 0.0

            edge_mapping = {
                y_edge.key: x_edge.key
                for x_edge, y_edge in zip(edges_x, edges_y)
                if x_edge and y_edge
            }

        total_similarity = (
            (node_similarity + edge_similarity) / 2.0
            if edge_similarity is not None
            else node_similarity
        )

        if self.normalize and alignment and len(alignment) > 0:
            total_similarity /= len(alignment)

        return GraphSim(
            value=total_similarity,
            node_mapping=node_mapping,
            edge_mapping=edge_mapping,
        )
