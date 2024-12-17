from dataclasses import dataclass
from .model import Graph, to_sequence, GraphSim
from ...helpers import dist2sim
from ...typing import AnySimFunc
from ..collections import dtw as dtwmodule

__all__ = ["dtw"]

@dataclass(slots=True, frozen=True)
class dtw:
    """
    Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

    Args:
        node_sim_func: A similarity function for nodes.
        edge_sim_func: An optional similarity function for edges.
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
        GraphSim(value=0.19444444444444442, node_mappings={'1': '1', '2': '2'}, edge_mappings={'e1': 'e1'})
    """

    node_sim_func: AnySimFunc
    edge_sim_func: AnySimFunc | None = None
    normalize: bool = True

    def __call__(
        self,
        x: Graph,
        y: Graph,
    ) -> GraphSim:
        sequence_x, edges_x = to_sequence(x)
        sequence_y, edges_y = to_sequence(y)

        # Use the dtwmodule for node distances
        node_result = dtwmodule(distance_func=self.node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )

        # Extract node distance and alignment
        node_distance = node_result.value
        alignment = node_result.local_similarities
        node_similarity = dist2sim(node_distance)

        node_mappings = {
            y_node.key: x_node.key for x_node, y_node in alignment if x_node and y_node
        }

        edge_similarity = None
        edge_mappings = {}

        if self.edge_sim_func:
            # Use the dtwmodule for edge distances
            edge_result = dtwmodule(distance_func=self.edge_sim_func)(
                edges_x, edges_y, return_alignment=False
            )

            edge_distance = edge_result.value
            edge_similarity = dist2sim(edge_distance)

            edge_mappings = {
                y_edge.key: x_edge.key
                for x_edge, y_edge in zip(edges_x, edges_y)
                if x_edge and y_edge
            }

        total_similarity = (
            (node_similarity + edge_similarity) / 2.0
            if edge_similarity is not None
            else node_similarity
        )

        if self.normalize and len(alignment) > 0:
            total_similarity /= len(alignment)

        return GraphSim(
            value=total_similarity,
            node_mappings=node_mappings,
            edge_mappings=edge_mappings,
        )
