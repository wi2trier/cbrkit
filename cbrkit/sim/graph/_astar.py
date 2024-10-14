from __future__ import annotations

import bisect
import itertools
import random
from collections.abc import Callable, Iterable
from dataclasses import InitVar, dataclass, field
from typing import Any, Literal, Protocol, override

from cbrkit.helpers import get_metadata, get_name, unpack_sims
from cbrkit.sim.graph._model import (
    Edge,
    Graph,
    Node,
)
from cbrkit.typing import (
    AnnotatedFloat,
    Casebase,
    Float,
    JsonDict,
    SimMapFunc,
    SimPairFunc,
    SupportsMetadata,
)


@dataclass(slots=True, frozen=True)
class GraphSim[K, N, E, G](AnnotatedFloat):
    value: float
    mapping: GraphMapping[K, N, E, G]


class HeuristicFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        s: SearchNode[K, N, E, G],
        /,
    ) -> float: ...


@dataclass(slots=True)
class GraphMapping[K, N, E, G]:
    """Store all mappings and perform integrity checks on them"""

    x: Graph[K, N, E, G]
    y: Graph[K, N, E, G]
    # mappings are from x to y
    node_mappings: dict[K, K] = field(default_factory=dict)
    edge_mappings: dict[K, K] = field(default_factory=dict)

    @property
    def unmapped_nodes(self) -> set[K]:
        """Return all unmapped nodes"""

        return set(self.y.nodes).difference(self.node_mappings.keys())

    @property
    def unmapped_edges(self) -> set[K]:
        """Return all unmapped edges"""

        return set(self.y.edges).difference(self.edge_mappings.keys())

    def _is_node_mapped(self, x: K) -> bool:
        """Check if given node is already mapped"""

        return x in self.node_mappings

    def _is_edge_mapped(self, x: K) -> bool:
        """Check if given edge is already mapped"""

        return x in self.edge_mappings

    def _are_nodes_mapped(self, x: K, y: K) -> bool:
        """Check if the two given nodes are mapped to each other"""

        return x in self.node_mappings and self.node_mappings[x] == y

    def is_legal_mapping(self, x: K, y: K) -> bool:
        """Check if mapping is legal"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            return self.is_legal_node_mapping(x, y)
        elif (y in self.y.edges) and (x in self.x.edges):
            return self.is_legal_edge_mapping(x, y)

        return False

    def is_legal_node_mapping(self, x: K, y: K) -> bool:
        """Check if mapping is legal"""

        return not (self._is_node_mapped(x) or type(x) is not type(y))

    def is_legal_edge_mapping(self, x: K, y: K) -> bool:
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

    def map(self, x: K, y: K) -> None:
        """Create a new mapping"""

        if (y in self.y.nodes) and (x in self.x.nodes):
            self.map_nodes(x, y)

        elif (y in self.y.edges) and (x in self.x.edges):
            self.map_edges(x, y)

    def map_nodes(self, x: K, y: K) -> None:
        """Create new node mapping"""

        self.node_mappings[x] = y

    def map_edges(self, x: K, y: K) -> None:
        """Create new edge mapping"""

        self.edge_mappings[x] = y


@dataclass(slots=True)
class SearchNode[K, N, E, G]:
    """Specific search node"""

    mapping: GraphMapping[K, N, E, G]
    f: float = 1.0
    unmapped_nodes: set[K] = field(init=False)
    unmapped_edges: set[K] = field(init=False)

    def __post_init__(self) -> None:
        # Initialize unmapped nodes and edges based on previous mapping
        self.unmapped_nodes = set(self.y.nodes.keys())
        self.unmapped_edges = set(self.y.edges.keys())

    @property
    def x(self) -> Graph[K, N, E, G]:
        return self.mapping.x

    @property
    def y(self) -> Graph[K, N, E, G]:
        return self.mapping.y

    def remove_unmapped_element(self, q: K) -> None:
        if q in self.unmapped_nodes:
            self.unmapped_nodes.remove(q)

        elif q in self.unmapped_edges:
            self.unmapped_edges.remove(q)


@dataclass(slots=True)
class astar[K, N, E, G, S: Float](
    SimMapFunc[
        Any,
        Graph[K, N, E, G],
        GraphSim[K, N, E, G],
    ],
    SupportsMetadata,
):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        node_sim_func: A similarity function for graph nodes
        edge_sim_func: A similarity function for graph edges
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.

    """

    node_sim_func: SimPairFunc[Node[K, N], S]
    edge_sim_func: SimPairFunc[Edge[K, N, E], S]
    queue_limit: int = 10000
    future_cost_func_name: InitVar[Literal["h1", "h2"]] = "h2"
    past_cost_func_name: InitVar[Literal["g1"]] = "g1"
    select_func_name: InitVar[Literal["select1"]] = "select1"
    future_cost_func: HeuristicFunc[K, N, E, G] = field(init=False)
    past_cost_func: HeuristicFunc[K, N, E, G] = field(init=False)
    select_func: Callable[[SearchNode[K, N, E, G]], None | tuple[K, Iterable[K]]] = (
        field(init=False)
    )

    def __post_init__(
        self,
        future_cost_func_name: str,
        past_cost_func_name: str,
        select_func_name: str,
    ) -> None:
        self.future_cost_func = getattr(self, future_cost_func_name)
        self.past_cost_func = getattr(self, past_cost_func_name)
        self.select_func = getattr(self, select_func_name)

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "node_sim_func": get_metadata(self.node_sim_func),
            "edge_sim_func": get_metadata(self.edge_sim_func),
            "queue_limit": self.queue_limit,
            "future_cost_func": get_name(self.future_cost_func),
            "past_cost_func": get_name(self.past_cost_func),
            "select_func": get_name(self.select_func),
        }

    def select1(
        self,
        s: SearchNode[K, N, E, G],
    ) -> (
        None
        | tuple[
            K,
            Iterable[K],
        ]
    ):
        query_obj = None
        candidates = None

        if s.unmapped_nodes:
            query_obj = random.choice(tuple(s.unmapped_nodes))
            candidates = s.x.nodes.keys()
        elif s.unmapped_edges:
            query_obj = random.choice(tuple(s.unmapped_edges))
            candidates = s.x.edges.keys()

        return (query_obj, candidates) if query_obj and candidates else None

    def h1(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        """Heuristic to compute future costs"""

        return (len(s.unmapped_nodes) + len(s.unmapped_edges)) / (
            len(s.y.nodes) + len(s.y.edges)
        )

    def h2(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        h_val = 0
        x, y = s.x, s.y

        for y_name in s.unmapped_nodes:
            max_sim = max(
                unpack_sims(
                    self.node_sim_func(x_node, y.nodes[y_name])
                    for x_node in (x.nodes.values())
                )
            )

            h_val += max_sim

        for y_name in s.unmapped_edges:
            max_sim = max(
                unpack_sims(
                    self.edge_sim_func(x_edge, y.edges[y_name])
                    for x_edge in x.edges.values()
                )
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))

    def g1(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        """Function to compute the costs of all previous steps"""

        node_sims = (
            self.node_sim_func(s.x.nodes[x], s.y.nodes[y])
            for x, y in s.mapping.node_mappings.items()
        )

        edge_sims = (
            self.edge_sim_func(s.x.edges[x], s.y.edges[y])
            for x, y in s.mapping.edge_mappings.items()
        )

        all_sims = unpack_sims(itertools.chain(node_sims, edge_sims))
        total_elements = len(s.y.nodes) + len(s.y.edges)

        return sum(all_sims) / total_elements

    def _expand(
        self,
        q: list[SearchNode[K, N, E, G]],
    ) -> list[SearchNode[K, N, E, G]]:
        """Expand a given node and its queue"""

        s = q[-1]
        mapped = False
        selected_objs = self.select_func(s)

        if selected_objs:
            query_obj, case_objs = selected_objs

            for case_obj in case_objs:
                if s.mapping.is_legal_mapping(query_obj, case_obj):
                    s_new = SearchNode(
                        s.mapping,
                    )
                    s_new.mapping.map(query_obj, case_obj)
                    s_new.remove_unmapped_element(query_obj)
                    s_new.f = self.past_cost_func(s_new) + self.future_cost_func(s_new)
                    bisect.insort(q, s_new, key=lambda x: x.f)
                    mapped = True

            if mapped:
                q.remove(s)
            else:
                s.remove_unmapped_element(query_obj)

        return q[len(q) - self.queue_limit :] if self.queue_limit > 0 else q

    def _astar_single(
        self,
        s0: SearchNode[K, N, E, G],
    ):
        """Perform an A* analysis of the x base and the y"""

        q = [s0]

        while q[-1].unmapped_nodes or q[-1].unmapped_edges:
            q = self._expand(q)

        return q[-1]

    @override
    def __call__(
        self,
        x_map: Casebase[K, Graph[K, N, E, G]],
        y: Graph[K, N, E, G],
    ) -> dict[K, GraphSim[K, N, E, G]]:
        results = {
            key: self._astar_single(SearchNode(GraphMapping(x, y)))
            for key, x in x_map.items()
        }

        return {
            key: GraphSim(self.past_cost_func(result), result.mapping)
            for key, result in results.items()
        }
