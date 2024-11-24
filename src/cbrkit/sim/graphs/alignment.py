from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, List
from ._model import Graph, Node, Edge
from cbrkit.typing import SimPairFunc, SimSeqFunc, Float
from ..collections import (
    dtw as dtw_func,
    smith_waterman as smith_waterman_func,
    mapping,
    isolated_mapping,
)

# Define type variables
K = TypeVar('K')  # Key type
N = TypeVar('N')  # Node data type
E = TypeVar('E')  # Edge data type
G = TypeVar('G')  # Graph data type


def get_outgoing_edges(
        graph: Graph[K, N, E, G],
        node: Node[K, N]
) -> List[Edge[K, N, E]]:
    """Get outgoing edges for a node in the graph.

    Args:
        graph: The graph containing the node
        node: The node to get outgoing edges for

    Returns:
        List of edges where the source is the given node
    """
    return [edge for edge in graph.edges.values() if edge.source.key == node.key]


def is_sequential_workflow(graph: Graph[K, N, E, G]) -> bool:
    """Check if the graph is a sequential workflow with a single directed path.

    A sequential workflow is defined as a graph where:
    - There is exactly one start node (node with no incoming edges).
    - There is exactly one end node (node with no outgoing edges).
    - Starting from the start node, there is a single path that visits all nodes.

    Args:
        graph: The graph to check

    Returns:
        True if the graph is a sequential workflow, False otherwise

    Examples:
        >>> # Creating a valid sequential workflow
        >>> from immutables import Map
        >>> from ._model import Graph, Node, Edge
        >>> nodes = Map({
        ...     "1": Node("1", "Step A"),
        ...     "2": Node("2", "Step B"),
        ...     "3": Node("3", "Step C")
        ... })
        >>> edges = Map({
        ...     "e1": Edge("e1", nodes["1"], nodes["2"], None),
        ...     "e2": Edge("e2", nodes["2"], nodes["3"], None)
        ... })
        >>> valid_graph = Graph(nodes, edges, None)
        >>> is_sequential_workflow(valid_graph)
        True

        >>> # Creating an invalid workflow (node with 2 outgoing edges)
        >>> edges_invalid = Map({
        ...     "e1": Edge("e1", nodes["1"], nodes["2"], None),
        ...     "e2": Edge("e2", nodes["1"], nodes["3"], None)
        ... })
        >>> invalid_graph = Graph(nodes, edges_invalid, None)
        >>> is_sequential_workflow(invalid_graph)
        False
    """
    if not graph.nodes:
        return False

    # Find nodes with no incoming edges (start nodes)
    incoming_edges = {node_key: 0 for node_key in graph.nodes}
    for edge in graph.edges.values():
        incoming_edges[edge.target.key] += 1

    start_nodes = [node for node_key, node in graph.nodes.items() if incoming_edges[node_key] == 0]

    if len(start_nodes) != 1:
        return False

    # Perform traversal from the start node
    visited_nodes = set()
    current_node = start_nodes[0]

    while True:
        visited_nodes.add(current_node.key)
        outgoing_edges = get_outgoing_edges(graph, current_node)
        if len(outgoing_edges) == 0:
            # End node reached
            break
        elif len(outgoing_edges) > 1:
            # Node has more than one outgoing edge
            return False
        else:
            # Move to the next node
            current_node = outgoing_edges[0].target

    # Check if all nodes were visited
    return len(visited_nodes) == len(graph.nodes)


def get_node_sequence(graph: Graph[K, N, E, G]) -> List[Node[K, N]]:
    """Extracts the sequence of nodes from a sequential workflow graph.

    Args:
        graph: The sequential workflow graph

    Returns:
        A list of nodes in the order of the workflow

    Raises:
        ValueError: If the graph is not a sequential workflow
    """
    if not is_sequential_workflow(graph):
        raise ValueError("Graph is not a sequential workflow")

    # Find the start node
    incoming_edges = {node_key: 0 for node_key in graph.nodes}
    for edge in graph.edges.values():
        incoming_edges[edge.target.key] += 1

    start_nodes = [node for node_key, node in graph.nodes.items() if incoming_edges[node_key] == 0]
    start_node = start_nodes[0]

    # Traverse the graph to get the node sequence
    node_sequence = []
    current_node = start_node

    while True:
        node_sequence.append(current_node)
        outgoing_edges = get_outgoing_edges(graph, current_node)
        if len(outgoing_edges) == 0:
            break
        else:
            current_node = outgoing_edges[0].target

    return node_sequence


@dataclass(slots=True, frozen=True)
class DynamicTimeWarpingAlignment(SimPairFunc[Graph[K, N, E, G], float]):
    """Performs Dynamic Time Warping alignment between two sequential workflows.

    This class implements DTW using mapping-based alignment for comparing
    sequential workflow graphs.

    Attributes:
        node_sim_func: Function to compute similarity between sequences of nodes

    Examples:
        >>> # Define a simple node similarity function
        >>> def node_sim(x: Node[K, N], y: Node[K, N]) -> float:
        ...     return 1.0 if x.data == y.data else 0.0
        >>> dtw_align = DynamicTimeWarpingAlignment(node_sim)

        >>> # Create two simple workflows using immutables.Map
        >>> from immutables import Map
        >>> from ._model import Graph, Node, Edge
        >>> nodes1 = Map({
        ...     "1": Node("1", "A"),
        ...     "2": Node("2", "B")
        ... })
        >>> edges1 = Map({
        ...     "e1": Edge("e1", nodes1["1"], nodes1["2"], None)
        ... })
        >>> graph1 = Graph(nodes1, edges1, None)

        >>> nodes2 = Map({
        ...     "1": Node("1", "A"),
        ...     "2": Node("2", "C")
        ... })
        >>> edges2 = Map({
        ...     "e1": Edge("e1", nodes2["1"], nodes2["2"], None)
        ... })
        >>> graph2 = Graph(nodes2, edges2, None)

        >>> dtw_align(graph1, graph2)  # Returns similarity score
        0.5
    """
    node_sim_func: SimSeqFunc[Node[K, N], Float]

    def _validate_inputs(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> None:
        """Validate that input graphs are sequential workflows."""
        if not (is_sequential_workflow(x) and is_sequential_workflow(y)):
            raise ValueError("Both graphs must be sequential workflows")

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform DTW alignment between two graphs."""
        self._validate_inputs(x, y)
        x_nodes = get_node_sequence(x)
        y_nodes = get_node_sequence(y)
        alignment_mapper = mapping(self.node_sim_func)
        alignment = alignment_mapper(x_nodes, y_nodes)
        return dtw_func()(alignment)


@dataclass(slots=True, frozen=True)
class SmithWatermanAlignment(SimPairFunc[Graph[K, N, E, G], float]):
    """Performs Smith-Waterman alignment between two sequential workflows.

    This class implements the Smith-Waterman algorithm using isolated mapping
    alignment for comparing sequential workflow graphs.

    Attributes:
        node_sim_func: Function to compute similarity between sequences of nodes

    Examples:
        >>> # Define a simple node similarity function
        >>> def node_sim(x: Node[K, N], y: Node[K, N]) -> float:
        ...     return 1.0 if x.data == y.data else 0.0
        >>> sw_align = SmithWatermanAlignment(node_sim)

        >>> # Create two simple workflows using immutables.Map
        >>> from immutables import Map
        >>> from ._model import Graph, Node, Edge
        >>> nodes1 = Map({
        ...     "1": Node("1", "A"),
        ...     "2": Node("2", "B"),
        ...     "3": Node("3", "C")
        ... })
        >>> edges1 = Map({
        ...     "e1": Edge("e1", nodes1["1"], nodes1["2"], None),
        ...     "e2": Edge("e2", nodes1["2"], nodes1["3"], None)
        ... })
        >>> graph1 = Graph(nodes1, edges1, None)

        >>> nodes2 = Map({
        ...     "1": Node("1", "A"),
        ...     "2": Node("2", "D"),
        ...     "3": Node("3", "C")
        ... })
        >>> edges2 = Map({
        ...     "e1": Edge("e1", nodes2["1"], nodes2["2"], None),
        ...     "e2": Edge("e2", nodes2["2"], nodes2["3"], None)
        ... })
        >>> graph2 = Graph(nodes2, edges2, None)

        >>> sw_align(graph1, graph2)  # Returns similarity score
        0.666667  # Higher score for matching subsequences A->*->C
    """
    node_sim_func: SimSeqFunc[Node[K, N], Float]

    def _validate_inputs(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> None:
        """Validate that input graphs are sequential workflows."""
        if not (is_sequential_workflow(x) and is_sequential_workflow(y)):
            raise ValueError("Both graphs must be sequential workflows")

    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform Smith-Waterman alignment between two graphs."""
        self._validate_inputs(x, y)
        x_nodes = get_node_sequence(x)
        y_nodes = get_node_sequence(y)
        isolated_align = isolated_mapping(self.node_sim_func)
        alignment = isolated_align(x_nodes, y_nodes)
        return smith_waterman_func()(alignment)


__all__ = ["DynamicTimeWarpingAlignment", "SmithWatermanAlignment", "is_sequential_workflow"]
