from __future__ import annotations

import bisect
import itertools
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, override

from cbrkit.helpers import (
    SimSeqWrapper,
    SimWrapper,
    get_metadata,
    unpack_sim,
    unpack_sims,
)
from cbrkit.sim.graph._model import (
    Edge,
    Graph,
    HasData,
    Node,
)
from cbrkit.typing import (
    AnnotatedFloat,
    AnySimFunc,
    Float,
    JsonDict,
    SimPairFunc,
    SimSeqFunc,
    SupportsMetadata,
)

type ElementKind = Literal["node", "edge"]


@dataclass(slots=True, frozen=True)
class GraphSim[K](AnnotatedFloat):
    value: float
    node_mappings: dict[K, K]
    edge_mappings: dict[K, K]


@dataclass(slots=True, frozen=True)
class SelectionResult[K]:
    query_element: K
    case_candidates: Iterable[K]
    kind: ElementKind


class SelectionFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        s: SearchNode[K, N, E, G],
        /,
    ) -> SelectionResult[K]: ...


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

    def is_legal_mapping(self, x: K, y: K, kind: ElementKind) -> bool:
        """Check if mapping is legal"""

        if kind == "node":
            return self.is_legal_node_mapping(x, y)
        elif kind == "edge":
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

    def map(self, x: K, y: K, kind: ElementKind) -> None:
        """Create a new mapping"""

        if kind == "node":
            self.map_nodes(x, y)

        elif kind == "edge":
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

    def remove_unmapped_element(self, q: K, kind: ElementKind) -> None:
        if kind == "node":
            self.unmapped_nodes.remove(q)

        elif kind == "edge":
            self.unmapped_edges.remove(q)


@dataclass(slots=True, frozen=True)
class default_edge_sim[K, N, E](SimSeqFunc[Edge[K, N, E], Float]):
    node_sim_func: SimSeqFunc[Node[K, N], Float]

    @override
    def __call__(
        self, pairs: Sequence[tuple[Edge[K, N, E], Edge[K, N, E]]]
    ) -> list[float]:
        source_sims = self.node_sim_func([(x.source, y.source) for x, y in pairs])
        target_sims = self.node_sim_func([(x.target, y.target) for x, y in pairs])

        return [
            0.5 * (unpack_sim(source) + unpack_sim(target))
            for source, target in zip(source_sims, target_sims, strict=True)
        ]


class DataSimWrapper[V: HasData[Any], S: Float](SimWrapper, SimSeqFunc[V, S]):
    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> Sequence[S]:
        if self.kind == "pair":
            func = cast(SimPairFunc[V, S], self.func)
            return [func(x.data, y.data) for (x, y) in pairs]

        func = cast(SimSeqFunc[V, S], self.func)
        return func([(x.data, y.data) for x, y in pairs])


@dataclass(slots=True)
class astar[K, N, E, G](
    SimPairFunc[Graph[K, N, E, G], GraphSim[K]],
    SupportsMetadata,
):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        node_obj_sim: A similarity function for graph nodes that receives two node objects.
        node_obj_sim: A similarity function for graph edges that receives two edge objects.
        node_data_sim: A similarity function for graph nodes that receives two data objects.
        edge_data_sim: A similarity function for graph edges that receives two data objects.
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.
        future_cost_func: A heuristic function to compute the future costs.
        past_cost_func: A heuristic function to compute the costs of all previous steps.
        select_func: A function to select the next element to map.

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    node_sim_func: SimSeqFunc[Node[K, N], Float]
    edge_sim_func: SimSeqFunc[Edge[K, N, E], Float]
    queue_limit: int
    future_cost_func: HeuristicFunc[K, N, E, G]
    past_cost_func: HeuristicFunc[K, N, E, G]
    select_func: SelectionFunc[K, N, E, G]

    def __init__(
        self,
        node_obj_sim: AnySimFunc[Node[K, N], Float] | None = None,
        edge_obj_sim: AnySimFunc[Edge[K, N, E], Float] | None = None,
        node_data_sim: AnySimFunc[N, Float] | None = None,
        edge_data_sim: AnySimFunc[E, Float] | None = None,
        queue_limit: int = 10000,
        future_cost_func: HeuristicFunc[K, N, E, G] | Literal["h1", "h2"] = "h2",
        past_cost_func: HeuristicFunc[K, N, E, G] | Literal["g1"] = "g1",
        select_func: SelectionFunc[K, N, E, G] | Literal["select1"] = "select1",
    ) -> None:
        # verify that only one of the object or data similarity functions is provided
        if (node_obj_sim and node_data_sim) or (edge_obj_sim and edge_data_sim):
            raise ValueError(
                "Only one of the object or data similarity functions can be provided for nodes and edges"
            )

        if node_data_sim:
            self.node_sim_func = DataSimWrapper(node_data_sim)
        elif node_obj_sim:
            self.node_sim_func = SimSeqWrapper(node_obj_sim)
        else:
            raise ValueError("Either node_obj_sim or node_data_sim must be provided")

        if edge_data_sim:
            self.edge_sim_func = DataSimWrapper(edge_data_sim)
        elif edge_obj_sim:
            self.edge_sim_func = SimSeqWrapper(edge_obj_sim)
        else:
            self.edge_sim_func = default_edge_sim(self.node_sim_func)

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

    def select1(
        self,
        s: SearchNode[K, N, E, G],
    ) -> None | SelectionResult[K]:
        if s.unmapped_nodes:
            return SelectionResult(
                query_element=random.choice(tuple(s.unmapped_nodes)),
                case_candidates=s.x.nodes.keys(),
                kind="node",
            )
        elif s.unmapped_edges:
            return SelectionResult(
                query_element=random.choice(tuple(s.unmapped_edges)),
                case_candidates=s.x.edges.keys(),
                kind="edge",
            )

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
                    self.node_sim_func(
                        [(x_node, y.nodes[y_name]) for x_node in (x.nodes.values())]
                    )
                )
            )

            h_val += max_sim

        for y_name in s.unmapped_edges:
            max_sim = max(
                unpack_sims(
                    self.edge_sim_func(
                        [(x_edge, y.edges[y_name]) for x_edge in x.edges.values()]
                    )
                )
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))

    def g1(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        """Function to compute the costs of all previous steps"""

        node_sims = self.node_sim_func(
            [(s.x.nodes[x], s.y.nodes[y]) for x, y in s.mapping.node_mappings.items()]
        )

        edge_sims = self.edge_sim_func(
            [(s.x.edges[x], s.y.edges[y]) for x, y in s.mapping.edge_mappings.items()]
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

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform an A* analysis of the x base and the y"""

        s0 = SearchNode(GraphMapping(x, y))
        q = [s0]

        while q[-1].unmapped_nodes or q[-1].unmapped_edges:
            q = self._expand(q)

        result = q[-1]

        return GraphSim(
            self.past_cost_func(result),
            result.mapping.node_mappings,
            result.mapping.edge_mappings,
        )
