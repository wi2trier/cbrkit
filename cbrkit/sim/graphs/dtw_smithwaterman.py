from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
from cbrkit.sim.graphs._model import Graph, Node
from cbrkit.typing import SimSeqFunc, Float
from ..collections import dtw as dtw_func, smith_waterman as smith_waterman_func, mapping, \
    isolated_mapping  # Import the existing functions and mappings

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
        """Perform DTW using existing dtw similarity function and mapping-based alignment."""
        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        # Use the mapping to generate the best alignment sequence for DTW
        alignment_mapping = mapping(self.node_sim_func)
        alignment_score = dtw_func()(alignment_mapping(x_nodes, y_nodes))

        return alignment_score

    def smith_waterman(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
        """Perform Smith-Waterman using existing smith_waterman similarity function and isolated_mapping alignment."""
        x_nodes = [self.node_sim_func([(node, node)])[0] for node in x.nodes.values()]
        y_nodes = [self.node_sim_func([(node, node)])[0] for node in y.nodes.values()]

        # Use isolated_mapping to generate the best alignment sequence for SMA
        isolated_alignment = isolated_mapping(self.node_sim_func)
        alignment_score = smith_waterman_func()(isolated_alignment(x_nodes, y_nodes))

        return alignment_score
