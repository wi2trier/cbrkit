import logging
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal, override

from ..._util import sorted_keys
from ...model.graph import (
    Graph,
)
from ...model.similarity import GraphSim
from ...schema import serializable
from ._base import PairSimilarities
from ._search import SearchGraphSimFunc, SearchState

__all__ = ["Greedy"]

logger = logging.getLogger(__name__)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Greedy[K: Hashable, N, E, G](SearchGraphSimFunc[K, N, E, G]):
    """Performs a Greedy search as described by [Dijkman et al. (2009)](https://doi.org/10.1007/978-3-642-03848-8_5)."""

    kind: Literal["greedy"] = field(default="greedy", repr=False)

    @override
    def search(
        self,
        case: Graph[K, N, E, G],
        query: Graph[K, N, E, G],
        sims: PairSimilarities[K],
        state: SearchState[K],
    ) -> GraphSim[K]:
        current_state = state
        current_sim = self.similarity(
            case,
            query,
            current_state.node_mapping,
            sims,
            edge_mapping=current_state.edge_mapping,
        )

        while not self.finished(current_state):
            # Iterate over all open pairs and find the best pair
            next_states: list[SearchState[K]] = []

            for y_key in sorted_keys(current_state.open_y_nodes):
                next_states.extend(self.expand_node(case, query, current_state, y_key))

            for y_key in sorted_keys(current_state.open_y_edges):
                next_states.extend(self.expand_edge(case, query, current_state, y_key))

            new_sims = [
                self.similarity(
                    case,
                    query,
                    state.node_mapping,
                    sims,
                    edge_mapping=state.edge_mapping,
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
