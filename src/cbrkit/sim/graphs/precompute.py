import itertools
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

from ...helpers import batchify_sim
from ...model.graph import (
    Graph,
    Node,
)
from ...typing import AnySimFunc, BatchSimFunc, Float
from ..wrappers import transpose_value
from .common import ElementMatcher


@dataclass(slots=True)
class precompute[K, N, E, G](BatchSimFunc[Graph[K, N, E, G], float]):
    node_sim_func: InitVar[AnySimFunc[N, Float]]
    node_matcher: ElementMatcher[N]
    batch_node_sim_func: BatchSimFunc[Node[K, N], Float] = field(init=False)
    # TODO: Add support for precomputing edges

    def __post_init__(self, any_node_sim_func: AnySimFunc[N, Float]) -> None:
        self.batch_node_sim_func = batchify_sim(transpose_value(any_node_sim_func))

    def __call__(
        self, batches: Sequence[tuple[Graph[K, N, E, G], Graph[K, N, E, G]]]
    ) -> list[float]:
        node_pairs: list[tuple[Node[K, N], Node[K, N]]] = []

        for x, y in batches:
            node_pairs.extend(
                (x.nodes[x_key], y.nodes[y_key])
                for x_key, y_key in itertools.product(x.nodes.keys(), y.nodes.keys())
                if self.node_matcher(x.nodes[x_key].value, y.nodes[y_key].value)
            )

        self.batch_node_sim_func(node_pairs)

        return [1.0] * len(batches)
