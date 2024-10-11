from __future__ import annotations

import bisect
import itertools
import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol

from cbrkit.helpers import unpack_sims
from cbrkit.sim.graph._model import (
    Edge,
    EdgeData,
    Graph,
    GraphData,
    Node,
    NodeData,
)
from cbrkit.typing import (
    Casebase,
    FloatProtocol,
    KeyType,
    SimMapFunc,
    SimPairFunc,
    SimType,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class GraphMapping(Generic[KeyType, NodeData, EdgeData, GraphData]):
    """Store all mappings and perform integrity checks on them"""

    x: Graph[KeyType, NodeData, EdgeData, GraphData]
    y: Graph[KeyType, NodeData, EdgeData, GraphData]
    # mappings are from x to y
    node_mappings: dict[KeyType, KeyType] = field(default_factory=dict)
    edge_mappings: dict[KeyType, KeyType] = field(default_factory=dict)

    @property
    def unmapped_nodes(self) -> set[KeyType]:
        """Return all unmapped nodes"""

        return set(self.y.nodes).difference(self.node_mappings.keys())

    @property
    def unmapped_edges(self) -> set[KeyType]:
        """Return all unmapped edges"""

        return set(self.y.edges).difference(self.edge_mappings.keys())

    def _is_node_mapped(self, x: KeyType) -> bool:
        """Check if given node is already mapped"""

        return x in self.node_mappings

    def _is_edge_mapped(self, x: KeyType) -> bool:
        """Check if given edge is already mapped"""

        return x in self.edge_mappings

    def _are_nodes_mapped(self, x: KeyType, y: KeyType) -> bool:
        """Check if the two given nodes are mapped to each other"""

        return x in self.node_mappings and self.node_mappings[x] == y

    def is_legal_mapping(self, x: KeyType, y: KeyType) -> bool:
        """Check if mapping is legal"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            return self.is_legal_node_mapping(x, y)
        elif (y in self.y.edges) and (x in self.x.edges):
            return self.is_legal_edge_mapping(x, y)

        return False

    def is_legal_node_mapping(self, x: KeyType, y: KeyType) -> bool:
        """Check if mapping is legal"""

        return not (self._is_node_mapped(x) or type(x) is not type(y))

    def is_legal_edge_mapping(self, x: KeyType, y: KeyType) -> bool:
        """Check if mapping is legal"""

        return not (
            self._is_edge_mapped(x)
            or not self.is_legal_node_mapping(
                self.y.edges[y].source.key, self.x.edges[x].source.key
            )
            or not self.is_legal_node_mapping(
                self.y.edges[y].target.key, self.x.edges[x].target.key
            )
        )

    def map(self, x: KeyType, y: KeyType) -> None:
        """Create a new mapping"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            self.map_nodes(x, y)

        elif (y in self.y.edges) and (x in self.x.edges):
            self.map_edges(x, y)

    def map_nodes(self, x: KeyType, y: KeyType) -> None:
        """Create new node mapping"""

        self.node_mappings[x] = y

    def map_edges(self, x: KeyType, y: KeyType) -> None:
        """Create new edge mapping"""

        self.edge_mappings[x] = y


@dataclass(slots=True)
class SearchNode(Generic[KeyType, NodeData, EdgeData, GraphData]):
    """Specific search node"""

    mapping: GraphMapping[KeyType, NodeData, EdgeData, GraphData]
    f: float = 1.0
    nodes: set[KeyType] = field(init=False)
    edges: set[KeyType] = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = set(self.y.nodes.keys())
        self.edges = set(self.y.edges.keys())

    @property
    def x(self) -> Graph[KeyType, NodeData, EdgeData, GraphData]:
        return self.mapping.x

    @property
    def y(self) -> Graph[KeyType, NodeData, EdgeData, GraphData]:
        return self.mapping.y

    def remove(self, q: KeyType) -> None:
        if q in self.nodes:
            self.nodes.remove(q)

        elif q in self.edges:
            self.edges.remove(q)


@dataclass(slots=True, frozen=True)
class GraphSim(FloatProtocol, Generic[KeyType, NodeData, EdgeData, GraphData]):
    value: float
    mapping: GraphMapping[KeyType, NodeData, EdgeData, GraphData]


class PastCostFunc(Protocol[KeyType, NodeData, EdgeData, GraphData]):
    def __call__(
        self,
        s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
        node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
        edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
        /,
    ) -> float: ...


class FutureCostFunc(Protocol[KeyType, NodeData, EdgeData, GraphData]):
    def __call__(
        self,
        s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
        x: Graph[KeyType, NodeData, EdgeData, GraphData],
        y: Graph[KeyType, NodeData, EdgeData, GraphData],
        node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
        edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
        /,
    ) -> float: ...


def select_func(
    s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
    x: Graph[KeyType, NodeData, EdgeData, GraphData],
) -> tuple[
    KeyType | None,
    Iterable[KeyType] | None,
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
    s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
    x: Graph[KeyType, NodeData, EdgeData, GraphData],
    y: Graph[KeyType, NodeData, EdgeData, GraphData],
    node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
    edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
) -> float:
    """Heuristic to compute future costs"""

    return (len(s.nodes) + len(s.edges)) / (len(y.nodes) + len(y.edges))


def h2(
    s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
    x: Graph[KeyType, NodeData, EdgeData, GraphData],
    y: Graph[KeyType, NodeData, EdgeData, GraphData],
    node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
    edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
) -> float:
    h_val = 0

    for y_name in s.nodes:
        max_sim = max(
            unpack_sims(
                node_sim_func(x_node, y.nodes[y_name]) for x_node in (x.nodes.values())
            )
        )

        h_val += max_sim

    for y_name in s.edges:
        max_sim = max(
            unpack_sims(
                edge_sim_func(x_edge, y.edges[y_name]) for x_edge in x.edges.values()
            )
        )

        h_val += max_sim

    return h_val / (len(y.nodes) + len(y.edges))


def g(
    s: SearchNode[KeyType, NodeData, EdgeData, GraphData],
    node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
    edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
) -> float:
    """Function to compute the costs of all previous steps"""

    node_sims = (
        node_sim_func(s.x.nodes[x], s.y.nodes[y])
        for x, y in s.mapping.node_mappings.items()
    )

    edge_sims = (
        edge_sim_func(s.x.edges[x], s.y.edges[y])
        for x, y in s.mapping.edge_mappings.items()
    )

    all_sims = unpack_sims(itertools.chain(node_sims, edge_sims))
    total_elements = len(s.mapping.y.nodes) + len(s.mapping.y.edges)

    return sum(all_sims) / total_elements


def astar(
    node_sim_func: SimPairFunc[Node[KeyType, NodeData], SimType],
    edge_sim_func: SimPairFunc[Edge[KeyType, NodeData, EdgeData], SimType],
    queue_limit: int = 10000,
    future_cost_func: FutureCostFunc[KeyType, NodeData, EdgeData, GraphData] = h2,
    past_cost_func: PastCostFunc[KeyType, NodeData, EdgeData, GraphData] = g,
) -> SimMapFunc[
    Any,
    Graph[KeyType, NodeData, EdgeData, GraphData],
    GraphSim[KeyType, NodeData, EdgeData, GraphData],
]:
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        node_sim_func: A similarity function for graph nodes
        edge_sim_func: A similarity function for graph edges
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.

    """

    def _expand(
        q: list[SearchNode[KeyType, NodeData, EdgeData, GraphData]],
        x: Graph[KeyType, NodeData, EdgeData, GraphData],
        y: Graph[KeyType, NodeData, EdgeData, GraphData],
    ) -> list[SearchNode[KeyType, NodeData, EdgeData, GraphData]]:
        """Expand a given node and its queue"""

        s = q[-1]
        mapped = False
        query_obj, iterator = select_func(s, x)

        if query_obj and iterator:
            for case_obj in iterator:
                if s.mapping.is_legal_mapping(query_obj, case_obj):
                    s_new = SearchNode(
                        s.mapping,
                    )
                    s_new.mapping.map(query_obj, case_obj)
                    s_new.remove(query_obj)
                    s_new.f = past_cost_func(
                        s_new, node_sim_func, edge_sim_func
                    ) + future_cost_func(s_new, x, y, node_sim_func, edge_sim_func)
                    bisect.insort(q, s_new, key=lambda x: x.f)
                    mapped = True

            if mapped:
                q.remove(s)
            else:
                s.remove(query_obj)

        return q[len(q) - queue_limit :] if queue_limit > 0 else q

    def _astar_single(
        x: Graph[KeyType, NodeData, EdgeData, GraphData],
        y: Graph[KeyType, NodeData, EdgeData, GraphData],
    ):
        """Perform an A* analysis of the x base and the y"""

        s0 = SearchNode(GraphMapping(x, y))
        q = [s0]

        while q[-1].nodes or q[-1].edges:
            q = _expand(q, x, y)

        return q[-1]

    def wrapped_func(
        x_map: Casebase[KeyType, Graph[KeyType, NodeData, EdgeData, GraphData]],
        y: Graph[KeyType, NodeData, EdgeData, GraphData],
    ) -> dict[KeyType, GraphSim[KeyType, NodeData, EdgeData, GraphData]]:
        results = {
            key: _astar_single(
                x,
                y,
            )
            for key, x in x_map.items()
        }

        return {
            key: GraphSim(g(result, node_sim_func, edge_sim_func), result.mapping)
            for key, result in results.items()
        }

    return wrapped_func
