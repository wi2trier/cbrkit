from __future__ import annotations

import bisect
import itertools
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, cast

from cbrkit.helpers import unpack_sims
from cbrkit.sim.graph._model import (
    EdgeData,
    EdgeKey,
    Graph,
    GraphData,
    NodeData,
    NodeKey,
)
from cbrkit.typing import Casebase, FloatProtocol, KeyType, SimPairFunc, SimType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphMapping(Generic[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]):
    """Store all mappings and perform integrity checks on them"""

    x: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]
    # mappings are from x to y
    node_mappings: dict[NodeKey, NodeKey] = field(default_factory=dict)
    edge_mappings: dict[EdgeKey, EdgeKey] = field(default_factory=dict)

    @property
    def unmapped_nodes(self) -> set[NodeKey]:
        """Return all unmapped nodes"""

        return set(self.y.nodes).difference(self.node_mappings.keys())

    @property
    def unmapped_edges(self) -> set[EdgeKey]:
        """Return all unmapped edges"""

        return set(self.y.edges).difference(self.edge_mappings.keys())

    def _is_node_mapped(self, x: NodeKey) -> bool:
        """Check if given node is already mapped"""

        return x in self.node_mappings

    def _is_edge_mapped(self, x: EdgeKey) -> bool:
        """Check if given edge is already mapped"""

        return x in self.edge_mappings

    def _are_nodes_mapped(self, x: NodeKey, y: NodeKey) -> bool:
        """Check if the two given nodes are mapped to each other"""

        return x in self.node_mappings and self.node_mappings[x] == y

    def is_legal_mapping(self, x: NodeKey | EdgeKey, y: NodeKey | EdgeKey) -> bool:
        """Check if mapping is legal"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            return self.is_legal_node_mapping(cast(NodeKey, x), cast(NodeKey, y))
        elif (y in self.y.edges) and (x in self.x.edges):
            return self.is_legal_edge_mapping(cast(EdgeKey, x), cast(EdgeKey, y))

        return False

    def is_legal_node_mapping(self, x: NodeKey, y: NodeKey) -> bool:
        """Check if mapping is legal"""

        return not (self._is_node_mapped(x) or type(x) is not type(y))

    def is_legal_edge_mapping(self, x: EdgeKey, y: EdgeKey) -> bool:
        """Check if mapping is legal"""

        return not (
            self._is_edge_mapped(x)
            or not self.is_legal_node_mapping(
                self.y.edges[y].source, self.x.edges[x].source
            )
            or not self.is_legal_node_mapping(
                self.y.edges[y].target, self.x.edges[x].target
            )
        )

    def map(self, x: NodeKey | EdgeKey, y: NodeKey | EdgeKey) -> None:
        """Create a new mapping"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            self.map_nodes(cast(NodeKey, x), cast(NodeKey, y))

        elif (y in self.y.edges) and (x in self.x.edges):
            self.map_edges(cast(EdgeKey, x), cast(EdgeKey, y))

    def map_nodes(self, x: NodeKey, y: NodeKey) -> None:
        """Create new node mapping"""

        self.node_mappings[x] = y

    def map_edges(self, x: EdgeKey, y: EdgeKey) -> None:
        """Create new edge mapping"""

        self.edge_mappings[x] = y


@dataclass(slots=True)
class SearchNode(Generic[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]):
    """Specific search node"""

    mapping: GraphMapping[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]
    f: float = 1.0
    nodes: set[NodeKey] = field(init=False)
    edges: set[EdgeKey] = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = set(self.y.nodes.keys())
        self.edges = set(self.y.edges.keys())

    @property
    def x(self) -> Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]:
        return self.mapping.x

    @property
    def y(self) -> Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]:
        return self.mapping.y

    def remove(self, q: NodeKey | EdgeKey) -> None:
        if q in self.nodes:
            self.nodes.remove(cast(NodeKey, q))

        elif q in self.edges:
            self.edges.remove(cast(EdgeKey, q))


@dataclass(slots=True, frozen=True)
class GraphSim(FloatProtocol, Generic[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]):
    value: float
    mapping: GraphMapping[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]


def astar(
    x_map: Casebase[KeyType, Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]],
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    node_sim_func: SimPairFunc[NodeData, SimType],
    edge_sim_func: SimPairFunc[EdgeData, SimType],
    queue_limit: int,
) -> dict[KeyType, GraphSim[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]]:
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        x_map: A casebase of graphs
        y: Query graph
        node_sim_func: A similarity function for graph nodes
        edge_sim_func: A similarity function for graph edges
        queue_limit: Limits the queue size which prunes the search space. This leads to a faster search and less memory usage but also introduces a similarity error.

    """

    results = {
        key: _astar_single(
            x,
            y,
            node_sim_func,
            edge_sim_func,
            queue_limit,
        )
        for key, x in x_map.items()
    }

    return {
        key: GraphSim(
            g(result, node_sim_func, edge_sim_func),
            result.mapping,
        )
        for key, result in results.items()
    }


# According to Bergmann and Gil, 2014
def _astar_single(
    x: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    node_sim_func: SimPairFunc[NodeData, SimType],
    edge_sim_func: SimPairFunc[EdgeData, SimType],
    queue_limit: int,
):
    """Perform an A* analysis of the x base and the y"""
    s0 = SearchNode(GraphMapping(x, y))
    q = [s0]

    while q[-1].nodes or q[-1].edges:
        q = _expand(q, x, y, node_sim_func, edge_sim_func, queue_limit)

    return q[-1]


def _expand(
    q: list[SearchNode[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]],
    x: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    node_sim_func: SimPairFunc[NodeData, SimType],
    edge_sim_func: SimPairFunc[EdgeData, SimType],
    queue_limit: int,
) -> list[SearchNode[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]]:
    """Expand a given node and its queue"""

    s = q[-1]
    mapped = False
    query_obj, iterator = select1(s, x)

    if query_obj and iterator:
        for case_obj in iterator:
            if s.mapping.is_legal_mapping(query_obj, case_obj):
                s_new = SearchNode(
                    s.mapping,
                )
                s_new.mapping.map(query_obj, case_obj)
                s_new.remove(query_obj)
                s_new.f = g(s_new, node_sim_func, edge_sim_func) + h2(
                    s_new, x, y, node_sim_func, edge_sim_func
                )
                bisect.insort(q, s_new, key=lambda x: x.f)
                mapped = True

        if mapped:
            q.remove(s)
        else:
            s.remove(query_obj)

    return q[len(q) - queue_limit :] if queue_limit > 0 else q


def select1(
    s: SearchNode[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    x: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
) -> tuple[
    NodeKey | EdgeKey | None,
    Iterable[NodeKey | EdgeKey] | None,
]:
    query_obj = None
    candidates = None

    if s.nodes:
        query_obj = random.choice(tuple(s.nodes))
        candidates = x.nodes.keys()
    elif s.edges:
        query_obj = random.choice(tuple(s.edges))
        candidates = x.edges.keys()

    return query_obj, candidates


def h1(
    s: SearchNode[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
) -> float:
    """Heuristic to compute future costs"""

    return (len(s.nodes) + len(s.edges)) / (len(y.nodes) + len(y.edges))


def h2(
    s: SearchNode[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    x: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    y: Graph[GraphData, NodeKey, NodeData, EdgeKey, EdgeData],
    node_sim_func: SimPairFunc[NodeData, SimType],
    edge_sim_func: SimPairFunc[EdgeData, SimType],
) -> float:
    h_val = 0

    for y_name in s.nodes:
        max_sim = max(
            unpack_sims(
                node_sim_func(x_node.data, y.nodes[y_name].data)
                for x_node in (x.nodes.values())
            )
        )

        h_val += max_sim

    for y_name in s.edges:
        max_sim = max(
            unpack_sims(
                edge_sim_func(x_edge.data, y.edges[y_name].data)
                for x_edge in x.edges.values()
            )
        )

        h_val += max_sim

    return h_val / (len(y.nodes) + len(y.edges))


def g(
    s: SearchNode[Any, Any, NodeData, Any, EdgeData],
    node_sim_func: SimPairFunc[NodeData, SimType],
    edge_sim_func: SimPairFunc[EdgeData, SimType],
) -> float:
    """Function to compute the costs of all previous steps"""

    node_sims = (
        node_sim_func(s.x.nodes[x].data, s.y.nodes[y].data)
        for x, y in s.mapping.node_mappings.items()
    )

    edge_sims = (
        edge_sim_func(s.x.edges[x].data, s.y.edges[y].data)
        for x, y in s.mapping.edge_mappings.items()
    )

    all_sims = unpack_sims(itertools.chain(node_sims, edge_sims))
    total_elements = len(s.mapping.y.nodes) + len(s.mapping.y.edges)

    return sum(all_sims) / total_elements
