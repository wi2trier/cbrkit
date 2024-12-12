from __future__ import annotations

import bisect
import itertools
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, override

import immutables

from ...helpers import (
    batchify_sim,
    unpack_float,
    unpack_floats,
)
from ...typing import (
    AnySimFunc,
    BatchSimFunc,
    Float,
    SimFunc,
)
from .model import (
    Edge,
    Graph,
    GraphSim,
    Node,
)

type ElementKind = Literal["node", "edge"]


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
    ) -> SelectionResult[K] | None: ...


class HeuristicFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        s: SearchNode[K, N, E, G],
        /,
    ) -> float: ...


@dataclass(slots=True, frozen=True)
class select1[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(self, s: SearchNode[K, N, E, G]) -> None | SelectionResult[K]:
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

        return None


@dataclass(slots=True, frozen=True)
class h1[K, N, E, G](HeuristicFunc[K, N, E, G]):
    def __call__(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        """Heuristic to compute future costs"""

        return (len(s.unmapped_nodes) + len(s.unmapped_edges)) / (
            len(s.y.nodes) + len(s.y.edges)
        )


@dataclass(slots=True)
class h2[K, N, E, G](HeuristicFunc[K, N, E, G]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[Node[K, N], Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(node_sim_func)

        if edge_sim_func is None:
            self.edge_sim_func = default_edge_sim(self.node_sim_func)
        else:
            self.edge_sim_func = batchify_sim(edge_sim_func)

    def __call__(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        h_val = 0

        for y_name in s.unmapped_nodes:
            max_sim = max(
                unpack_floats(
                    self.node_sim_func(
                        [(x_node, s.y.nodes[y_name]) for x_node in (s.x.nodes.values())]
                    )
                )
            )

            h_val += max_sim

        for y_name in s.unmapped_edges:
            max_sim = max(
                unpack_floats(
                    self.edge_sim_func(
                        [(x_edge, s.y.edges[y_name]) for x_edge in s.x.edges.values()]
                    )
                )
            )

            h_val += max_sim

        return h_val / (len(s.y.nodes) + len(s.y.edges))


@dataclass(slots=True)
class g1[K, N, E, G](HeuristicFunc[K, N, E, G]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]
    edge_sim_func: BatchSimFunc[Edge[K, N, E], Float]

    def __init__(
        self,
        node_sim_func: AnySimFunc[Node[K, N], Float],
        edge_sim_func: AnySimFunc[Edge[K, N, E], Float] | None = None,
    ) -> None:
        self.node_sim_func = batchify_sim(node_sim_func)

        if edge_sim_func is None:
            self.edge_sim_func = default_edge_sim(self.node_sim_func)
        else:
            self.edge_sim_func = batchify_sim(edge_sim_func)

    def __call__(
        self,
        s: SearchNode[K, N, E, G],
    ) -> float:
        """Function to compute the costs of all previous steps"""

        node_sims = self.node_sim_func(
            [(s.x.nodes[x], s.y.nodes[y]) for y, x in s.node_mappings.items()]
        )

        edge_sims = self.edge_sim_func(
            [(s.x.edges[x], s.y.edges[y]) for y, x in s.edge_mappings.items()]
        )

        all_sims = unpack_floats(itertools.chain(node_sims, edge_sims))
        total_elements = len(s.y.nodes) + len(s.y.edges)

        return sum(all_sims) / total_elements


@dataclass(slots=True, frozen=True)
class SearchNode[K, N, E, G]:
    """Specific search node"""

    x: Graph[K, N, E, G]
    y: Graph[K, N, E, G]
    # mappings are from y to x
    node_mappings: immutables.Map[K, K]
    edge_mappings: immutables.Map[K, K]
    unmapped_nodes: frozenset[K]
    unmapped_edges: frozenset[K]

    def is_legal_mapping(self, x: K, y: K, kind: ElementKind) -> bool:
        """Check if mapping is legal"""

        if kind == "node":
            return self.is_legal_node_mapping(x, y)
        elif kind == "edge":
            return self.is_legal_edge_mapping(x, y)

    def is_legal_node_mapping(self, x: K, y: K) -> bool:
        """Check if mapping is legal"""

        return not (y in self.node_mappings or type(x) is not type(y))

    def is_legal_edge_mapping(self, x: K, y: K) -> bool:
        """Check if mapping is legal"""

        return not (
            y in self.edge_mappings
            or not self.is_legal_node_mapping(
                self.x.edges[x].source.key, self.y.edges[y].source.key
            )
            or not self.is_legal_node_mapping(
                self.x.edges[x].target.key, self.y.edges[y].target.key
            )
        )


type ScoredSearchNode[K, N, E, G] = tuple[SearchNode[K, N, E, G], float]


@dataclass(slots=True, frozen=True)
class default_edge_sim[K, N, E](BatchSimFunc[Edge[K, N, E], Float]):
    node_sim_func: BatchSimFunc[Node[K, N], Float]

    @override
    def __call__(
        self, batches: Sequence[tuple[Edge[K, N, E], Edge[K, N, E]]]
    ) -> list[float]:
        source_sims = self.node_sim_func([(x.source, y.source) for x, y in batches])
        target_sims = self.node_sim_func([(x.target, y.target) for x, y in batches])

        return [
            0.5 * (unpack_float(source) + unpack_float(target))
            for source, target in zip(source_sims, target_sims, strict=True)
        ]


@dataclass(slots=True)
class astar[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        past_cost_func: A heuristic function to compute the costs of all previous steps.
        future_cost_func: A heuristic function to compute the future costs.
        select_func: A function to select the next element to map.
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    past_cost_func: HeuristicFunc[K, N, E, G]
    future_cost_func: HeuristicFunc[K, N, E, G]
    select_func: SelectionFunc[K, N, E, G]
    queue_limit: int = 10000

    def _expand(
        self,
        q: list[ScoredSearchNode[K, N, E, G]],
    ) -> list[ScoredSearchNode[K, N, E, G]]:
        """Expand a given node and its queue"""

        s, s_score = q.pop(-1)
        selection = self.select_func(s)
        mapped: bool = False

        if selection:
            for case_element in selection.case_candidates:
                if s.is_legal_mapping(
                    selection.query_element, case_element, selection.kind
                ):
                    s_new = SearchNode(
                        s.x,
                        s.y,
                        s.node_mappings.set(selection.query_element, case_element)
                        if selection.kind == "node"
                        else s.node_mappings,
                        s.edge_mappings.set(selection.query_element, case_element)
                        if selection.kind == "edge"
                        else s.edge_mappings,
                        s.unmapped_nodes - {selection.query_element}
                        if selection.kind == "node"
                        else s.unmapped_nodes,
                        s.unmapped_edges - {selection.query_element}
                        if selection.kind == "edge"
                        else s.unmapped_edges,
                    )

                    s_new_scored = (
                        s_new,
                        self.past_cost_func(s_new) + self.future_cost_func(s_new),
                    )
                    bisect.insort(q, s_new_scored, key=lambda x: x[1])
                    mapped = True

            if not mapped:
                s_new = SearchNode(
                    s.x,
                    s.y,
                    s.node_mappings,
                    s.edge_mappings,
                    s.unmapped_nodes - {selection.query_element}
                    if selection.kind == "node"
                    else s.unmapped_nodes,
                    s.unmapped_edges - {selection.query_element}
                    if selection.kind == "edge"
                    else s.unmapped_edges,
                )
                s_new_scored = (
                    s_new,
                    self.past_cost_func(s_new) + self.future_cost_func(s_new),
                )
                bisect.insort(q, s_new_scored, key=lambda x: x[1])

        return q[len(q) - self.queue_limit :] if self.queue_limit > 0 else q

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform an A* analysis of the x base and the y"""

        s0: ScoredSearchNode[K, N, E, G] = (
            SearchNode(
                x,
                y,
                immutables.Map(),
                immutables.Map(),
                frozenset(x.nodes.keys()),
                frozenset(x.edges.keys()),
            ),
            1.0,
        )
        q = [s0]

        while q[-1][0].unmapped_nodes or q[-1][0].unmapped_edges:
            q = self._expand(q)

        result, _ = q[-1]

        return GraphSim(
            self.past_cost_func(result),
            result.node_mappings,
            result.edge_mappings,
        )
