from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple, Any
from ._model import Graph, Node, is_sequential
from ...helpers import dist2sim

__all__ = []

try:
    import numpy as np
    from ..collections import dtw

    @dataclass(slots=True)
    class GraphDTW:
        """
        Graph-based Dynamic Time Warping similarity function leveraging sequence alignment.

        Examples:
            >>> # Example node and edge similarity functions
            >>> node_similarity = lambda n1, n2: 1.0 if n1.data == n2.data else 0.0
            >>> edge_similarity = lambda e1, e2: 1.0 if e1.data == e2.data else 0.0
            >>> # Create example graphs
            >>> from src.cbrkit.sim.graphs._model import Graph, Node, Edge
            >>> import immutables
            >>> nodes1 = immutables.Map({
            ...     '1': Node(key='1', data='A'),
            ...     '2': Node(key='2', data='B'),
            ... })
            >>> edges1 = immutables.Map({
            ...     'e1': Edge(key='e1', source=nodes1['1'], target=nodes1['2'], data='X'),
            ... })
            >>> graph1 = Graph(nodes=nodes1, edges=edges1, data=None)
            >>> nodes2 = immutables.Map({
            ...     '1': Node(key='1', data='A'),
            ...     '2': Node(key='2', data='C'),
            ... })
            >>> edges2 = immutables.Map({
            ...     'e1': Edge(key='e1', source=nodes2['1'], target=nodes2['2'], data='Y'),
            ... })
            >>> graph2 = Graph(nodes=nodes2, edges=edges2, data=None)
            >>> # Instantiate GraphDTW
            >>> g_dtw = GraphDTW(node_similarity, edge_similarity)
            >>> result = g_dtw(graph1, graph2)
            >>> print(result)
            (0.3333333333333333, [('1', '1'), ('2', '2')])
        """

        node_sim_func: Callable[[Any, Any], float]
        edge_sim_func: Callable[[Any, Any], float] | None = None  # Optional for edge similarity
        normalize: bool = True  # Whether to normalize similarity by sequence length

        def __call__(
                self,
                graph1: Graph[K, N, E, G],
                graph2: Graph[K, N, E, G]
        ) -> Tuple[float, List[Tuple[K, K]]]:
            """
            Perform Graph-DTW using node and optional edge similarity functions.

            Args:
                graph1: The first graph.
                graph2: The second graph.

            Returns:
                A tuple of (similarity score, best alignment).
            """
            # Check if the graphs are sequential
            if not (is_sequential(graph1) and is_sequential(graph2)):
                raise ValueError("Input graphs must be sequential workflows")

            # Extract sequences of nodes in order
            sequence1 = self.get_sequential_nodes(graph1)
            sequence2 = self.get_sequential_nodes(graph2)

            # Convert similarity function to distance function
            node_distance_func = lambda a, b: 1.0 - self.node_sim_func(a, b)

            # Use the dtw module to compute alignment based on node distances
            dtw_instance = dtw(distance_func=node_distance_func)
            node_distance, alignment = dtw_instance(sequence1, sequence2, return_alignment=True)

            # Convert node distance to similarity
            node_similarity = dist2sim(node_distance)

            # Optionally compute edge similarity if edge similarity function is provided
            if self.edge_sim_func:
                # Extract sequences of edges in order
                edge_sequence1 = self.get_sequential_edges(graph1, sequence1)
                edge_sequence2 = self.get_sequential_edges(graph2, sequence2)

                # Convert similarity function to distance function
                edge_distance_func = lambda a, b: 1.0 - self.edge_sim_func(a, b)

                # Use the dtw module to compute alignment based on edge distances
                dtw_edge_instance = dtw(distance_func=edge_distance_func)
                edge_distance = dtw_edge_instance(edge_sequence1, edge_sequence2, return_alignment=False)

                # Convert edge distance to similarity
                edge_similarity = dist2sim(edge_distance)

                # Combine node and edge similarities (average)
                total_similarity = (node_similarity + edge_similarity) / 2.0
            else:
                total_similarity = node_similarity

            # Optionally normalize the similarity score by the length of the alignment path
            if self.normalize and len(alignment) > 0:
                total_similarity /= len(alignment)

            # Return node keys in the alignment for clarity
            alignment_keys = [(a.key if a else None, b.key if b else None) for a, b in alignment]

            return total_similarity, alignment_keys

        def get_sequential_nodes(self, graph: Graph[K, N, E, G]) -> List[Node[K, N]]:
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
            start_nodes = [node for node in graph.nodes.values() if in_degree[node.key] == 0]
            if len(start_nodes) != 1:
                raise ValueError("Graph does not have a unique start node")
            start_node = start_nodes[0]

            sequence = []
            current_node = start_node
            visited_nodes = set()
            while current_node and current_node.key not in visited_nodes:
                sequence.append(current_node)
                visited_nodes.add(current_node.key)
                outgoing_edges = [edge for edge in graph.edges.values() if edge.source.key == current_node.key]
                if len(outgoing_edges) > 1:
                    raise ValueError("Graph is not sequential (node has multiple outgoing edges)")
                current_node = outgoing_edges[0].target if outgoing_edges else None
            return sequence

        def get_sequential_edges(self, graph: Graph[K, N, E, G], node_sequence: List[Node[K, N]]) -> List[Edge[K, N, E]]:
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
                    (edge for edge in graph.edges.values()
                     if edge.source.key == source_key and edge.target.key == target_key),
                    None
                )
                if edge:
                    edges.append(edge)
                else:
                    raise ValueError(f"No edge found between {source_key} and {target_key}")
            return edges

        __all__ += ["graph_dtw"]
except ImportError:
    pass
