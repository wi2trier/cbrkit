from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from frozendict import frozendict

from ...helpers import (
    get_logger,
)
from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import GraphSim, SearchGraphSimFunc, SearchState

logger = get_logger(__name__)

__all__ = ["greedy"]


@dataclass(slots=True)
class greedy[K, N, E, G](
    SearchGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """
    Performs a Greedy search as described by [Dijkman et al. (2009)](https://doi.org/10.1007/978-3-642-03848-8_5).

    Args:
        node_sim_func: A function to compute the similarity between two nodes.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_sim_func: A function to compute the similarity between two edges.
        edge_matcher: A function that returns true if two edges can be mapped legally.
        start_with: A string indicating whether to start with nodes or edges.

    Returns:
        The similarity between the query graph and the most similar graph in the casebase.
    """

    start_with: Literal["nodes", "edges"] = "nodes"

    def expand_edges(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
    ) -> list[SearchState[K]]:
        """Expand the current state by adding all possible edge mappings"""

        next_states: list[SearchState[K]] = []

        for y_key in state.open_edges:
            next_states.extend(self.expand_edge(x, y, state, y_key))

        return next_states

    def expand_nodes(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        state: SearchState[K],
    ) -> list[SearchState[K]]:
        """Expand the current state by adding all possible node mappings"""

        next_states: list[SearchState[K]] = []

        for y_key in state.open_nodes:
            next_states.extend(self.expand_node(x, y, state, y_key))

        return next_states

    def expand(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
        current_state: SearchState[K],
        current_sim: GraphSim[K],
        expand_func: Callable[
            [Graph[K, N, E, G], Graph[K, N, E, G], SearchState[K]], list[SearchState[K]]
        ],
        node_pair_sims: Mapping[tuple[K, K], float],
        edge_pair_sims: Mapping[tuple[K, K], float],
    ) -> tuple[SearchState[K], GraphSim[K]]:
        """Expand the current state by adding all possible mappings"""

        while current_state.open_nodes or current_state.open_edges:
            # Iterate over all open pairs and find the best pair
            next_states = expand_func(x, y, current_state)
            new_sims = [
                self.similarity(
                    x,
                    y,
                    state.mapped_nodes,
                    state.mapped_edges,
                    node_pair_sims,
                    edge_pair_sims,
                )
                for state in next_states
            ]

            best_sim, best_state = max(
                zip(new_sims, next_states, strict=True),
                key=lambda item: item[0].value,
            )

            # If no better pair is found, break the loop
            if best_sim.value <= current_sim.value:
                current_state = SearchState(
                    current_state.mapped_nodes,
                    current_state.mapped_edges,
                    frozenset(),
                    frozenset(),
                )
                break

            # Update the current state and similarity
            current_state = best_state
            current_sim = best_sim

        return current_state, current_sim

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        """Perform greedy graph matching of the query y against the case x"""

        state = SearchState(
            frozendict(),
            frozendict(),
            frozenset(y.nodes.keys()),
            frozenset(y.edges.keys()),
        )
        sim = GraphSim(0.0, frozendict(), frozendict(), frozendict(), frozendict())

        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        if self.start_with == "nodes":
            state, sim = self.expand(
                x, y, state, sim, self.expand_nodes, node_pair_sims, edge_pair_sims
            )
            state, sim = self.expand(
                x, y, state, sim, self.expand_edges, node_pair_sims, edge_pair_sims
            )

        elif self.start_with == "edges":
            state, sim = self.expand(
                x, y, state, sim, self.expand_edges, node_pair_sims, edge_pair_sims
            )
            state, sim = self.expand(
                x, y, state, sim, self.expand_nodes, node_pair_sims, edge_pair_sims
            )

        else:
            raise ValueError(
                f"Invalid start_with value: {self.start_with}. Expected 'nodes' or 'edges'."
            )

        return sim
