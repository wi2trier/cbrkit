from __future__ import annotations

import bisect
import itertools
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, override

import rustworkx

from cbrkit.helpers import get_metadata, unpack_sims
from cbrkit.typing import (
    AnnotatedFloat,
    Casebase,
    Float,
    JsonDict,
    SimMapFunc,
    SimPairFunc,
    SupportsMetadata,
)

type Graph[V, E] = rustworkx.PyDiGraph[V, E]

type ElementKind = Literal["node", "edge"]


@dataclass(slots=True, frozen=True)
class GraphSim[V, E](AnnotatedFloat):
    value: float
    mapping: GraphMapping[V, E]


@dataclass(slots=True, frozen=True)
class SelectionResult:
    query_element: int
    case_candidates: Iterable[int]
    kind: ElementKind


class SelectionFunc[V, E](Protocol):
    def __call__(
        self,
        s: SearchNode[V, E],
        /,
    ) -> SelectionResult: ...


class HeuristicFunc[V, E](Protocol):
    def __call__(
        self,
        s: SearchNode[V, E],
        /,
    ) -> float: ...


@dataclass(slots=True)
class GraphMapping[V, E]:
    """Store all mappings and perform integrity checks on them"""

    x: Graph[V, E]
    y: Graph[V, E]
    # mappings are from x to y
    node_mappings: dict[int, int] = field(default_factory=dict)
    edge_mappings: dict[int, int] = field(default_factory=dict)

    @property
    def unmapped_nodes(self) -> set[int]:
        """Return all unmapped nodes"""

        return set(self.y.node_indices()).difference(self.node_mappings.keys())

    @property
    def unmapped_edges(self) -> set[int]:
        """Return all unmapped edges"""

        return set(self.y.edge_indices()).difference(self.edge_mappings.keys())

    def _is_node_mapped(self, x: int) -> bool:
        """Check if given node is already mapped"""

        return x in self.node_mappings

    def _is_edge_mapped(self, x: int) -> bool:
        """Check if given edge is already mapped"""

        return x in self.edge_mappings

    def _are_nodes_mapped(self, x: int, y: int) -> bool:
        """Check if the two given nodes are mapped to each other"""

        return x in self.node_mappings and self.node_mappings[x] == y

    def is_legal_mapping(self, x: int, y: int, kind: ElementKind) -> bool:
        """Check if mapping is legal"""

        if kind == "node":
            return self.is_legal_node_mapping(x, y)
        elif kind == "edge":
            return self.is_legal_edge_mapping(x, y)

        return False

    def is_legal_node_mapping(self, x: int, y: int) -> bool:
        """Check if mapping is legal"""

        return not (self._is_node_mapped(x) or type(x) is not type(y))

    def is_legal_edge_mapping(self, x: int, y: int) -> bool:
        """Check if mapping is legal"""

        x_source, x_target = self.x.get_edge_endpoints_by_index(x)
        y_source, y_target = self.y.get_edge_endpoints_by_index(y)

        return not (
            self._is_edge_mapped(x)
            or not self.is_legal_node_mapping(y_source, x_source)
            or not self.is_legal_node_mapping(y_target, x_target)
        )

    def map(self, x: int, y: int, kind: ElementKind) -> None:
        """Create a new mapping"""

        if kind == "node":
            self.map_nodes(x, y)

        elif kind == "edge":
            self.map_edges(x, y)

    def map_nodes(self, x: int, y: int) -> None:
        """Create new node mapping"""

        self.node_mappings[x] = y

    def map_edges(self, x: int, y: int) -> None:
        """Create new edge mapping"""

        self.edge_mappings[x] = y


@dataclass(slots=True)
class SearchNode[V, E]:
    """Specific search node"""

    mapping: GraphMapping[V, E]
    f: float = 1.0
    unmapped_nodes: set[int] = field(init=False)
    unmapped_edges: set[int] = field(init=False)

    def __post_init__(self) -> None:
        # Initialize unmapped nodes and edges based on previous mapping
        self.unmapped_nodes = set(self.y.node_indices())
        self.unmapped_edges = set(self.y.edge_indices())

    @property
    def x(self) -> Graph[V, E]:
        return self.mapping.x

    @property
    def y(self) -> Graph[V, E]:
        return self.mapping.y

    def remove_unmapped_element(self, q: int, type: ElementKind) -> None:
        if type == "node":
            self.unmapped_nodes.remove(q)

        elif type == "edge":
            self.unmapped_edges.remove(q)


@dataclass(slots=True)
class astar[K, V, E](
    SimMapFunc[Any, Graph[V, E], GraphSim[V, E]],
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

    node_sim_func: SimPairFunc[V, Float]
    edge_sim_func: SimPairFunc[E, Float]
    queue_limit: int
    future_cost_func: HeuristicFunc[V, E]
    past_cost_func: HeuristicFunc[V, E]
    select_func: SelectionFunc[V, E]

    def __init__(
        self,
        node_sim_func: SimPairFunc[V, Float],
        edge_sim_func: SimPairFunc[E, Float] | Literal["default"] = "default",
        queue_limit: int = 10000,
        future_cost_func: HeuristicFunc[V, E] | Literal["h1", "h2"] = "h2",
        past_cost_func: HeuristicFunc[V, E] | Literal["g1"] = "g1",
        select_func: SelectionFunc[V, E] | Literal["select1"] = "select1",
    ) -> None:
        self.node_sim_func = node_sim_func
        self.edge_sim_func = (
            self.default_edge_sim if edge_sim_func == "default" else edge_sim_func
        )
        self.queue_limit = queue_limit
        self.future_cost_func = (
            getattr(self, future_cost_func)
            if isinstance(future_cost_func, str)
            else future_cost_func
        )
        self.past_cost_func = (
            getattr(self, past_cost_func)
            if isinstance(past_cost_func, str)
            else past_cost_func
        )
        self.select_func = (
            getattr(self, select_func) if isinstance(select_func, str) else select_func
        )

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "node_sim_func": get_metadata(self.node_sim_func),
            "edge_sim_func": get_metadata(self.edge_sim_func),
            "queue_limit": self.queue_limit,
            "future_cost_func": get_metadata(self.future_cost_func),
            "past_cost_func": get_metadata(self.past_cost_func),
            "select_func": get_metadata(self.select_func),
        }

    def default_edge_sim(self, x: E, y: E) -> float:
        return 0.0  # TODO

    def select1(
        self,
        s: SearchNode[V, E],
    ) -> None | SelectionResult:
        if s.unmapped_nodes:
            return SelectionResult(
                query_element=random.choice(tuple(s.unmapped_nodes)),
                case_candidates=s.x.node_indices(),
                kind="node",
            )
        elif s.unmapped_edges:
            return SelectionResult(
                query_element=random.choice(tuple(s.unmapped_edges)),
                case_candidates=s.x.edge_indices(),
                kind="edge",
            )

    def h1(
        self,
        s: SearchNode[V, E],
    ) -> float:
        """Heuristic to compute future costs"""

        return (len(s.unmapped_nodes) + len(s.unmapped_edges)) / (
            s.y.num_nodes() + s.y.num_edges()
        )

    def h2(
        self,
        s: SearchNode[V, E],
    ) -> float:
        h_val = 0
        x, y = s.x, s.y

        for y_id in s.unmapped_nodes:
            y_data = y.get_node_data(y_id)
            max_sim = max(
                unpack_sims(
                    self.node_sim_func(x.get_node_data(x_id), y_data)
                    for x_id in x.node_indices()
                )
            )

            h_val += max_sim

        for y_id in s.unmapped_edges:
            y_data = y.get_edge_data_by_index(y_id)
            max_sim = max(
                unpack_sims(
                    self.edge_sim_func(x.get_edge_data_by_index(x_id), y_data)
                    for x_id in x.edge_indices()
                )
            )

            h_val += max_sim

        return h_val / (y.num_nodes() + y.num_edges())

    def g1(
        self,
        s: SearchNode[V, E],
    ) -> float:
        """Function to compute the costs of all previous steps"""

        node_sims = (
            self.node_sim_func(s.x.get_node_data(x), s.y.get_node_data(y))
            for x, y in s.mapping.node_mappings.items()
        )

        edge_sims = (
            self.edge_sim_func(
                s.x.get_edge_data_by_index(x), s.y.get_edge_data_by_index(y)
            )
            for x, y in s.mapping.edge_mappings.items()
        )

        all_sims = unpack_sims(itertools.chain(node_sims, edge_sims))
        total_elements = s.y.num_nodes() + s.y.num_edges()

        return sum(all_sims) / total_elements

    def _expand(
        self,
        q: list[SearchNode[V, E]],
    ) -> list[SearchNode[V, E]]:
        """Expand a given node and its queue"""

        s = q[-1]
        mapped = False
        selection = self.select_func(s)

        if selection:
            for case_element in selection.case_candidates:
                if s.mapping.is_legal_mapping(
                    selection.query_element, case_element, selection.kind
                ):
                    s_new = SearchNode(s.mapping)
                    s_new.mapping.map(
                        selection.query_element, case_element, selection.kind
                    )
                    s_new.remove_unmapped_element(
                        selection.query_element, selection.kind
                    )
                    s_new.f = self.past_cost_func(s_new) + self.future_cost_func(s_new)
                    bisect.insort(q, s_new, key=lambda x: x.f)
                    mapped = True

            if mapped:
                q.remove(s)
            else:
                s.remove_unmapped_element(selection.query_element, selection.kind)

        return q[len(q) - self.queue_limit :] if self.queue_limit > 0 else q

    def _astar_single(
        self,
        s0: SearchNode[V, E],
    ):
        """Perform an A* analysis of the x base and the y"""

        q = [s0]

        while q[-1].unmapped_nodes or q[-1].unmapped_edges:
            q = self._expand(q)

        return q[-1]

    @override
    def __call__(
        self,
        x_map: Casebase[K, Graph[V, E]],
        y: Graph[V, E],
    ) -> dict[K, GraphSim[V, E]]:
        results = {
            key: self._astar_single(SearchNode(GraphMapping(x, y)))
            for key, x in x_map.items()
        }

        return {
            key: GraphSim(self.past_cost_func(result), result.mapping)
            for key, result in results.items()
        }
