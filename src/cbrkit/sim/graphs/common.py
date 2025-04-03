from dataclasses import dataclass
from typing import Any, Protocol

from ...typing import StructuredValue


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
