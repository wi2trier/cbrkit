from dataclasses import dataclass
from .model import Graph, to_sequence, GraphSim
from ...helpers import dist2sim
from ...typing import AnySimFunc, BatchSimFunc,SimFunc
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
        GraphSim(value=0.05555555555555555, node_mappings={'1': '1', '2': '2'}, edge_mappings={'e1': 'e1'})

        >>> # Example with BatchSimFunc
        >>> def batch_node_similarity(pairs):
        ...     return [1.0 if n1.value == n2.value else 0.0 for n1, n2 in pairs]
        >>> def batch_edge_similarity(pairs):
        ...     return [1.0 if e1.value == e2.value else 0.0 for e1, e2 in pairs]
        >>> g_dtw_batch = dtw(batch_node_similarity, batch_edge_similarity)
        >>> result_batch = g_dtw_batch(graph_x, graph_y)
        >>> print(result_batch)
        GraphSim(value=0.05555555555555555, node_mappings={'1': '1', '2': '2'}, edge_mappings={'e1': 'e1'})
    """

    node_sim_func: AnySimFunc
    edge_sim_func: AnySimFunc | None = None
    normalize: bool = True

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        sequence_x, edges_x = to_sequence(x)
        sequence_y, edges_y = to_sequence(y)

        # Wrap BatchSimFunc if necessary
        wrapped_node_sim_func = self._wrap_batch_func(self.node_sim_func)
        wrapped_edge_sim_func = (
            self._wrap_batch_func(self.edge_sim_func)
            if self.edge_sim_func is not None
            else None
        )

        # Use the dtwmodule for node distances
        node_result = dtwmodule(distance_func=wrapped_node_sim_func)(
            sequence_x, sequence_y, return_alignment=True
        )

        # Extract alignment and node distance
        alignment = node_result.mapping
        node_distance = node_result.value
        node_similarity = self._compute_similarity_from_func(
            self.node_sim_func, alignment
        )

        node_mappings = {
            y_node.key: x_node.key for x_node, y_node in alignment if x_node and y_node
        }

        edge_similarity = None
        edge_mappings = {}

        if self.edge_sim_func:
            # Use the dtwmodule for edge distances
            edge_result = dtwmodule(distance_func=wrapped_edge_sim_func)(
                edges_x, edges_y, return_alignment=False
            )

            edge_distance = edge_result.value
            edge_similarity = self._compute_similarity_from_func(
                self.edge_sim_func, zip(edges_x, edges_y)
            )

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

        if self.normalize and alignment and len(alignment) > 0:
            total_similarity /= len(alignment)

        return GraphSim(
            value=total_similarity,
            node_mappings=node_mappings,
            edge_mappings=edge_mappings,
        )

    def _wrap_batch_func(self, func: AnySimFunc) -> AnySimFunc:
        """
        Wrap a BatchSimFunc to behave like a SimFunc for pairwise comparisons.

        Args:
            func: The similarity function (SimFunc or BatchSimFunc).

        Returns:
            A function that can handle pairwise comparisons.
        """
        def wrapper(a, b):
            try:
                return func([(a, b)])[0]  # Call the batch function with one pair
            except TypeError:
                return func(a, b)  # Treat as SimFunc if batch fails
        return wrapper

    def _compute_similarity_from_func(
        self, sim_func: AnySimFunc, pairs: Sequence[tuple]
    ) -> float:
        """
        Compute the similarity from either a SimFunc or BatchSimFunc.

        Args:
            sim_func: The similarity function (SimFunc or BatchSimFunc).
            pairs: A sequence of pairs to compare.

        Returns:
            The combined similarity score as a float.
        """
        try:
            # Try treating it as a BatchSimFunc by passing the whole batch
            similarities = sim_func(pairs)
            if isinstance(similarities, Sequence):  # If it returns a sequence, treat it as BatchSimFunc
                return sum(similarities) / len(similarities) if similarities else 0.0
        except TypeError:
            # Fallback to treating it as a SimFunc
            similarities = [sim_func(a, b) for a, b in pairs]
            return sum(similarities) / len(similarities) if similarities else 0.0
