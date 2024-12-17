from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
from .model import Graph, Node, Edge, is_sequential, GraphSim
from ...helpers import dist2sim
from ...typing import SimFunc
from ..collections import dtw as dtwmodule

__all__ = ["dtw"]


@dataclass(slots=True, frozen=True)
class dtw[K](SimFunc[Graph[K, Any, Any, Any], GraphSim[K]]):
    """
    Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

    Examples:
        >>> node_similarity = lambda n1, n2: 1.0 if n1.value == n2.value else 0.0
        >>> edge_similarity = lambda e1, e2: 1.0 if e1.value == e2.value else 0.0
        >>> import immutables
        >>> nodes_x = immutables.Map({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='B'),
        ... })
        >>> edges_x = immutables.Map({
        ...     'e1': Edge(key='e1', source=nodes_x['1'], target=nodes_x['2'], value='X'),
        ... })
        >>> graph_x = Graph(nodes=nodes_x, edges=edges_x, value=None)
        >>> nodes_y = immutables.Map({
        ...     '1': Node(key='1', value='A'),
        ...     '2': Node(key='2', value='C'),
        ... })
        >>> edges_y = immutables.Map({
        ...     'e1': Edge(key='e1', source=nodes_y['1'], target=nodes_y['2'], value='Y'),
        ... })
        >>> graph_y = Graph(nodes=nodes_y, edges=edges_y, value=None)
        >>> g_dtw = dtw(node_similarity, edge_similarity)
        >>> result = g_dtw(graph_x, graph_y)
        >>> print(result)
        GraphSim(value=0.19444444444444442, node_mappings={'1': '1', '2': '2'}, edge_mappings={'e1': 'e1'})
    """

    node_sim_func: Callable[[Any, Any], float]
    edge_sim_func: Callable[[Any, Any], float] | None = None
    normalize: bool = True

    def __call__(
        self,
        x: Graph[K, Any, Any, Any],
        y: Graph[K, Any, Any, Any],
    ) -> GraphSim[K]:
        """
        Perform Graph-DTW using node and optional edge similarity functions.

        Args:
            x: The case graph.
            y: The query graph.

        Returns:
            A GraphSim object containing the similarity score and mappings.
        """
        if not (is_sequential(x) and is_sequential(y)):
            raise ValueError("Input graphs must be sequential workflows")

        sequence_x = self.get_sequential_nodes(x)
        sequence_y = self.get_sequential_nodes(y)

        # Use the dtwmodule for node distances
        node_result = dtwmodule(distance_func=self.node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )

        # Extract node distance (value) and alignment (local_similarities)
        node_distance = node_result.value
        alignment = node_result.local_similarities

        node_similarity = dist2sim(node_distance)

        node_mappings = {
            y_node.key: x_node.key for x_node, y_node in alignment if x_node and y_node
        }

        edge_similarity = None
        edge_mappings = {}

        if self.edge_sim_func:
            edge_sequence_x = self.get_sequential_edges(x, sequence_x)
            edge_sequence_y = self.get_sequential_edges(y, sequence_y)

            # Use the dtwmodule for edge distances
            edge_result = dtwmodule(distance_func=self.edge_sim_func)(
                edge_sequence_x, edge_sequence_y, return_alignment=False
            )

            # Extract edge distance (value)
            edge_distance = edge_result.value

            edge_similarity = dist2sim(edge_distance)

            edge_mappings = {
                y_edge.key: x_edge.key
                for x_edge, y_edge in zip(edge_sequence_x, edge_sequence_y)
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

    def get_sequential_nodes(
        self, graph: Graph[K, Any, Any, Any]
    ) -> list[Node[K, Any]]:
        """
        Retrieves the nodes of the graph in sequential order.

        Args:
            graph: The graph to extract nodes from.

        Returns:
            A list of nodes in sequential order.
        """
        in_degree = {node.key: 0 for node in graph.nodes.values()}
        for edge in graph.edges.values():
            in_degree[edge.target.key] += 1
        start_nodes = [
            node for node in graph.nodes.values() if in_degree[node.key] == 0
        ]
        if len(start_nodes) != 1:
            raise ValueError("Graph does not have a unique start node")
        start_node = start_nodes[0]

        sequence = []
        current_node = start_node
        visited_nodes = set()
        while current_node and current_node.key not in visited_nodes:
            sequence.append(current_node)
            visited_nodes.add(current_node.key)
            outgoing_edges = [
                edge
                for edge in graph.edges.values()
                if edge.source.key == current_node.key
            ]
            if len(outgoing_edges) > 1:
                raise ValueError(
                    "Graph is not sequential (node has multiple outgoing edges)"
                )
            current_node = outgoing_edges[0].target if outgoing_edges else None
        return sequence

    def get_sequential_edges(
        self, graph: Graph[K, Any, Any, Any], node_sequence: list[Node[K, Any]]
    ) -> list[Edge[K, Any, Any]]:
        """
        Retrieves the edges of the graph in sequential order based on node sequence.

        Args:
            graph: The graph to extract edges from.
            node_sequence: The list of nodes in sequential order.

        Returns:
            A list of edges in sequential order.
        """
        edges = []
        for i in range(len(node_sequence) - 1):
            source_key = node_sequence[i].key
            target_key = node_sequence[i + 1].key
            edge = next(
                (
                    edge
                    for edge in graph.edges.values()
                    if edge.source.key == source_key and edge.target.key == target_key
                ),
                None,
            )
            if edge:
                edges.append(edge)
            else:
                raise ValueError(f"No edge found between {source_key} and {target_key}")
        return edges
