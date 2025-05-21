from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override

from ...helpers import unpack_float
from ...model.graph import Edge, Node
from ...typing import BatchSimFunc, Float, StructuredValue


@dataclass(slots=True, frozen=True)
class GraphSim[K](StructuredValue[float]):
    node_mapping: dict[K, K]
    edge_mapping: dict[K, K]
    node_similarities: dict[K, float]
    edge_similarities: dict[K, float]


class ElementMatcher[T](Protocol):
    def __call__(self, x: T, y: T, /) -> bool: ...


def default_element_matcher(x: Any, y: Any) -> bool:
    return True


def type_element_matcher(x: Any, y: Any) -> bool:
    return type(x) is type(y)


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
