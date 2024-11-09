from __future__ import annotations
from dataclasses import dataclass
from typing import Generic
from cbrkit.helpers import SimSeqFunc
from cbrkit.sim.graphs._model import Graph, Node 
from cbrkit.typing import Float

from ..collections import dtw as dtw_func, \
    smith_waterman as smith_waterman_func  # Import the existing DTW and SW implementations


@dataclass(slots=True)
class GraphAlignment[K, N, E, G](Generic[K, N, E, G]):
    """
    A class to perform graph alignment algorithms, specifically DTW and Smith-Waterman, on sequential workflows.

    Args:
        node_sim_func: A similarity function for graph nodes.
    """
    node_sim_func: SimSeqFunc[Node[K, N], Float]

    def is_sequential_workflow(self, graph: Graph[K, N, E, G]) -> bool:
        """Check if the graph is a sequential workflow with a single directed path."""
        nodes = list(graph.nodes.values())
        if not nodes:
            return False

        # Check that each node (except the last) has exactly one outgoing edge and the last has none
        for node in nodes[:-1]:
            if len(graph.get_outgoing_edges(node)) != 1:
                return False
        return len(graph.get_outgoing_edges(nodes[-1])) == 0

    def dynamic_time_warping(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform DTW using the existing dtw similarity function."""
        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        # Use the existing dtw function from collections.py
        return dtw_func()(x_nodes, y_nodes)

    def smith_waterman(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform Smith-Waterman using the existing smith_waterman similarity function."""
        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        # Use the existing smith_waterman function from collections.py
        return smith_waterman_func()(x_nodes, y_nodes)
