from __future__ import annotations

import heapq
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, override

import immutables

from ...helpers import batchify_sim, unpack_float, unpack_floats
from ...typing import AnySimFunc, BatchSimFunc, Float, SimFunc
from .model import Edge, ElementType, Graph, GraphSim, Node

__all__ = [
    "HeuristicFunc",
    "SelectionFunc",
    "ElementMatcher",
    "h1",
    "h2",
    "g1",
    "select1",
    "build",
]


class HeuristicFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        /,
    ) -> float: ...


class SelectionFunc[K, N, E, G](Protocol):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
        /,
    ) -> None | tuple[K, ElementType]: ...


class ElementMatcher[T](Protocol):
    def __call__(self, x: T, y: T, /) -> bool: ...


@dataclass(slots=True, frozen=True)
class h1[K, N, E, G](HeuristicFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        """Heuristic to compute future costs"""

        return (len(s.remaining_nodes) + len(s.remaining_edges)) / (
            len(y.nodes) + len(y.edges)
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
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        h_val = 0

        for y_name in s.remaining_nodes:
            max_sim = max(
                unpack_floats(
                    self.node_sim_func(
                        [(x_node, y.nodes[y_name]) for x_node in (x.nodes.values())]
                    )
                )
            )

            h_val += max_sim

        for y_name in s.remaining_edges:
            max_sim = max(
                unpack_floats(
                    self.edge_sim_func(
                        [(x_edge, y.edges[y_name]) for x_edge in x.edges.values()]
                    )
                )
            )

            h_val += max_sim

        return h_val / (len(y.nodes) + len(y.edges))


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
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> float:
        """Function to compute the costs of all previous steps"""

        node_sims = self.node_sim_func(
            [
                (x.nodes[x_key], y.nodes[y_key])
                for y_key, x_key in s.mapped_nodes.items()
            ]
        )

        edge_sims = self.edge_sim_func(
            [
                (x.edges[x_key], y.edges[y_key])
                for y_key, x_key in s.mapped_edges.items()
            ]
        )

        all_sims = unpack_floats(itertools.chain(node_sims, edge_sims))
        total_elements = len(y.nodes) + len(y.edges)

        return sum(all_sims) / total_elements


class select1[K, N, E, G](SelectionFunc[K, N, E, G]):
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        s: State[K],
    ) -> None | tuple[K, ElementType]:
        """Select the next node or edge to be mapped"""

        if s.remaining_nodes:
            return next(iter(s.remaining_nodes)), "node"

        if s.remaining_edges:
            return next(iter(s.remaining_edges)), "edge"

        return None


@dataclass(slots=True, frozen=True)
class State[K]:
    # mappings are from y to x
    mapped_nodes: immutables.Map[K, K]
    mapped_edges: immutables.Map[K, K]
    remaining_nodes: frozenset[K]
    remaining_edges: frozenset[K]


@dataclass(slots=True, frozen=True, order=True)
class PriorityState[K]:
    priority: float
    state: State[K] = field(compare=False)


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


def default_element_matcher(x: Any, y: Any) -> bool:
    return type(x) is type(y)


@dataclass(slots=True)
class build[K, N, E, G](SimFunc[Graph[K, N, E, G], GraphSim[K]]):
    """
    Performs the A* algorithm proposed by [Bergmann and Gil (2014)](https://doi.org/10.1016/j.is.2012.07.005) to compute the similarity between a query graph and the graphs in the casebase.

    Args:
        past_cost_func: A heuristic function to compute the costs of all previous steps.
        future_cost_func: A heuristic function to compute the future costs.
        selection_func: A function to select the next node or edge to be mapped.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        queue_limit: Limits the queue size which prunes the search space.
            This leads to a faster search and less memory usage but also introduces a similarity error.

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    past_cost_func: HeuristicFunc[K, N, E, G]
    future_cost_func: HeuristicFunc[K, N, E, G]
    selection_func: SelectionFunc[K, N, E, G]
    node_matcher: ElementMatcher[N] = default_element_matcher
    edge_matcher: ElementMatcher[E] = default_element_matcher
    queue_limit: int = 10000

    def expand_node(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        y_key: K,
    ) -> list[State[K]]:
        """Expand a given node and its queue"""

        next_states: list[State[K]] = []
        y_value = y.nodes[y_key]

        for x_key in x.nodes.keys():
            x_value = x.nodes[x_key]

            if (
                y_key not in state.mapped_nodes.keys()
                and x_key not in state.mapped_nodes.values()
                and self.node_matcher(x_value.value, y_value.value)
            ):
                next_states.append(
                    State(
                        state.mapped_nodes.set(y_key, x_key),
                        state.mapped_edges,
                        state.remaining_nodes - {y_key},
                        state.remaining_edges,
                    )
                )

        if not next_states:
            next_states.append(
                State(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.remaining_nodes - {y_key},
                    state.remaining_edges,
                )
            )

        return next_states

    def expand_edge(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
        y_key: K,
    ) -> list[State[K]]:
        """Expand a given edge and its queue"""

        next_states: list[State[K]] = []
        y_value = y.edges[y_key]

        for x_key in x.edges.keys():
            x_value = x.edges[x_key]

            if (
                y_key not in state.mapped_edges.keys()
                and x_key not in state.mapped_edges.values()
                and self.edge_matcher(x_value.value, y_value.value)
            ):
                mapped_x_source_key = state.mapped_nodes.get(y_value.source.key)
                mapped_x_target_key = state.mapped_nodes.get(y_value.target.key)

                # if the nodes are already mapped, check if they are mapped legally
                if (
                    mapped_x_source_key is not None
                    and x_value.source.key != mapped_x_source_key
                ) or (
                    mapped_x_target_key is not None
                    and x_value.target.key != mapped_x_target_key
                ):
                    continue

                # optimization: if the nodes are not mapped yet, map them since this is required for a legal mapping
                mapped_nodes = state.mapped_nodes
                remaining_nodes = state.remaining_nodes

                if y_value.source.key not in mapped_nodes:
                    mapped_nodes = mapped_nodes.set(
                        y_value.source.key, x_value.source.key
                    )
                    remaining_nodes -= {y_value.source.key}

                if y_value.target.key not in mapped_nodes:
                    mapped_nodes = mapped_nodes.set(
                        y_value.target.key, x_value.target.key
                    )
                    remaining_nodes -= {y_value.target.key}

                next_states.append(
                    State(
                        mapped_nodes,
                        state.mapped_edges.set(y_key, x_key),
                        remaining_nodes,
                        state.remaining_edges - {y_key},
                    )
                )

        if not next_states:
            next_states.append(
                State(
                    state.mapped_nodes,
                    state.mapped_edges,
                    state.remaining_nodes,
                    state.remaining_edges - {y_key},
                )
            )

        return next_states

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: State[K],
    ) -> list[State[K]]:
        """Expand a given node and its queue"""

        selection = self.selection_func(x, y, state)

        if selection is None:
            return []

        y_key, y_type = selection

        if y_type == "node":
            return self.expand_node(x, y, state, y_key)

        if y_type == "edge":
            return self.expand_edge(x, y, state, y_key)

        raise ValueError(f"Unknown element type: {y_type}")

    @override
    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform an A* analysis of the x base and the y"""

        open_set: list[PriorityState[K]] = []
        best_sim = 0.0
        best_state = State(
            immutables.Map(),
            immutables.Map(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
        )
        heapq.heappush(open_set, PriorityState(1 - best_sim, best_state))

        while open_set:
            first_elem = heapq.heappop(open_set)
            current_sim = 1 - first_elem.priority
            current_state = first_elem.state

            if current_sim > best_sim:
                best_sim = current_sim
                best_state = current_state

            for next_state in self.expand(x, y, current_state):
                next_sim = self.past_cost_func(
                    x, y, next_state
                ) + self.future_cost_func(x, y, next_state)
                heapq.heappush(open_set, PriorityState(1 - next_sim, next_state))

            if self.queue_limit > 0 and len(open_set) > self.queue_limit:
                open_set = open_set[: self.queue_limit]

        return GraphSim(
            best_sim,
            best_state.mapped_nodes,
            best_state.mapped_edges,
        )
