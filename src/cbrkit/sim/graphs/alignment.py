from __future__ import annotations
from dataclasses import dataclass
from typing import override
from cbrkit.sim.graphs._model import Graph, Node
from cbrkit.typing import SimPairFunc, SimSeqFunc, Float
from ..collections import (
    dtw as dtw_func,
    smith_waterman as smith_waterman_func,
    mapping,
    isolated_mapping,
)

def is_sequential_workflow[K, N, E, G](graph: Graph[K, N, E, G]) -> bool:
    """Check if the graph is a sequential workflow with a single directed path."""
    nodes = list(graph.nodes.values())
    if not nodes:
        return False

    # Check that each node (except the last) has exactly one outgoing edge and the last has none
    for node in nodes[:-1]:
        if len(graph.get_outgoing_edges(node)) != 1:
            return False
    return len(graph.get_outgoing_edges(nodes[-1])) == 0


@dataclass(slots=True, frozen=True)
class DynamicTimeWarpingAlignment[K, N, E, G](SimPairFunc[Graph[K, N, E, G], float]):
    node_sim_func: SimSeqFunc[Node[K, N], Float]
    """
    Performs Dynamic Time Warping alignment on sequential workflows.

    Example:
        >>> from cbrkit.sim.graphs._model import Graph, Node
        >>> # Create two simple sequential graphs
        >>> g1 = Graph()
        >>> g1.add_node(Node("1", "A"))
        >>> g1.add_node(Node("2", "B"))
        >>> g1.add_edge(g1.nodes["1"], g1.nodes["2"], None)
        >>> g2 = Graph()
        >>> g2.add_node(Node("1", "A"))
        >>> g2.add_node(Node("2", "B"))
        >>> g2.add_edge(g2.nodes["1"], g2.nodes["2"], None)
        >>> # Create mock similarity function
        >>> def mock_sim(pairs): return [1.0 if n1.data == n2.data else 0.0 for n1, n2 in pairs]
        >>> dtw = DynamicTimeWarpingAlignment(mock_sim)
        >>> dtw(g1, g2)
        2.0
    """

    @override
    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform DTW using mapping-based alignment."""
        if not (is_sequential_workflow(x) and is_sequential_workflow(y)):
            raise ValueError("Both graphs must be sequential workflows")

        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        alignment_mapping = mapping(self.node_sim_func)
        return dtw_func()(alignment_mapping(x_nodes, y_nodes))


@dataclass(slots=True, frozen=True)
class SmithWatermanAlignment[K, N, E, G](SimPairFunc[Graph[K, N, E, G], float]):
    node_sim_func: SimSeqFunc[Node[K, N], Float]
    """
    Performs Smith-Waterman alignment on sequential workflows.

    Example:
        >>> from cbrkit.sim.graphs._model import Graph, Node
        >>> # Create two simple sequential graphs
        >>> g1 = Graph()
        >>> g1.add_node(Node("1", "A"))
        >>> g1.add_node(Node("2", "B"))
        >>> g1.add_edge(g1.nodes["1"], g1.nodes["2"], None)
        >>> g2 = Graph()
        >>> g2.add_node(Node("1", "A"))
        >>> g2.add_node(Node("2", "C"))
        >>> g2.add_edge(g2.nodes["1"], g2.nodes["2"], None)
        >>> # Create mock similarity function
        >>> def mock_sim(pairs): return [1.0 if n1.data == n2.data else 0.0 for n1, n2 in pairs]
        >>> swa = SmithWatermanAlignment(mock_sim)
        >>> swa(g1, g2)
        1.0
    """

    @override
    def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform Smith-Waterman using isolated_mapping alignment."""
        if not (is_sequential_workflow(x) and is_sequential_workflow(y)):
            raise ValueError("Both graphs must be sequential workflows")

        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        isolated_align = isolated_mapping(self.node_sim_func)
        return smith_waterman_func()(isolated_align(x_nodes, y_nodes))


__all__ = ["DynamicTimeWarpingAlignment", "SmithWatermanAlignment", "is_sequential_workflow"]