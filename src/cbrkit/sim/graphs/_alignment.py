from __future__ import annotations

from dataclasses import dataclass
from typing import override

from ...typing import Float, SimPairFunc, SimSeqFunc
from ._model import Graph, Node, is_sequential

__all__ = []

try:
    from ..collections import dtw as dtw_func
    from ..collections import mapping

    @dataclass(slots=True, frozen=True)
    class dtw[K, N, E, G](SimPairFunc[Graph[K, N, E, G], float]):
        """
        Performs Dynamic Time Warping alignment on sequential workflows.

        Examples:
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

        node_sim_func: SimSeqFunc[Node[K, N], Float]

        @override
        def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
            """Perform DTW using mapping-based alignment."""
            if not (is_sequential(x) and is_sequential(y)):
                raise ValueError("Both graphs must be sequential workflows")

            x_nodes = [
                self.node_sim_func([(node, node)])[0] for node in x.nodes.values()
            ]
            y_nodes = [
                self.node_sim_func([(node, node)])[0] for node in y.nodes.values()
            ]

            alignment_mapping = mapping(self.node_sim_func)
            return dtw_func()(alignment_mapping(x_nodes, y_nodes))

    __all__ += ["dtw"]

except ImportError:
    pass

try:
    from ..collections import isolated_mapping
    from ..collections import smith_waterman as smith_waterman_func

    @dataclass(slots=True, frozen=True)
    class smith_waterman[K, N, E, G](SimPairFunc[Graph[K, N, E, G], float]):
        """
        Performs Smith-Waterman alignment on sequential workflows.

        Examples:
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
            >>> swa = smith_waterman(mock_sim)
            >>> swa(g1, g2)
            1.0
        """

        node_sim_func: SimSeqFunc[Node[K, N], Float]

        @override
        def __call__(self, x: Graph[K, N, E, G], y: Graph[K, N, E, G]) -> float:
            """Perform Smith-Waterman using isolated_mapping alignment."""
            if not (is_sequential(x) and is_sequential(y)):
                raise ValueError("Both graphs must be sequential workflows")

            x_nodes = [
                self.node_sim_func([(node, node)])[0] for node in x.nodes.values()
            ]
            y_nodes = [
                self.node_sim_func([(node, node)])[0] for node in y.nodes.values()
            ]

            isolated_align = isolated_mapping(self.node_sim_func)
            return smith_waterman_func()(isolated_align(x_nodes, y_nodes))

    __all__ += ["smith_waterman"]

except ImportError:
    pass
