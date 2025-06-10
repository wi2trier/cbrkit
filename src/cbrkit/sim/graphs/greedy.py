from dataclasses import dataclass

from ...helpers import (
    get_logger,
)
from ...model.graph import (
    Graph,
)
from ...typing import SimFunc
from .common import GraphSim, SearchGraphSimFunc, SearchState

logger = get_logger(__name__)


@dataclass(slots=True)
class greedy[K, N, E, G](
    SearchGraphSimFunc[K, N, E, G], SimFunc[Graph[K, N, E, G], GraphSim[K]]
):
    """Performs a Greedy search as described by [Dijkman et al. (2009)](https://doi.org/10.1007/978-3-642-03848-8_5)

    Args:
        node_sim_func: A function to compute the similarity between two nodes.
        edge_sim_func: A function to compute the similarity between two edges.
        node_matcher: A function that returns true if two nodes can be mapped legally.
        edge_matcher: A function that returns true if two edges can be mapped legally.

    Returns:
        The similarity between a query and a case graph along with the mapping.
    """

    def __call__(
        self,
        x: Graph[K, N, E, G],
        y: Graph[K, N, E, G],
    ) -> GraphSim[K]:
        node_pair_sims, edge_pair_sims = self.pair_similarities(x, y)

        current_state = self.init_search_state(x, y)
        current_sim = self.similarity(
            x,
            y,
            current_state.node_mapping,
            current_state.edge_mapping,
            node_pair_sims,
            edge_pair_sims,
        )

        while not self.finished(current_state):
            # Iterate over all open pairs and find the best pair
            next_states: list[SearchState[K]] = []

            for y_key in current_state.open_y_nodes:
                next_states.extend(self.expand_node(x, y, current_state, y_key))

            for y_key in current_state.open_y_edges:
                next_states.extend(self.expand_edge(x, y, current_state, y_key))

            new_sims = [
                self.similarity(
                    x,
                    y,
                    state.node_mapping,
                    state.edge_mapping,
                    node_pair_sims,
                    edge_pair_sims,
                )
                for state in next_states
            ]

            best_sim, best_state = max(
                zip(new_sims, next_states, strict=True),
                key=lambda item: item[0].value,
            )

            # Update the current state and similarity
            current_state = best_state
            current_sim = best_sim

        return current_sim
