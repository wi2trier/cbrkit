from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np

from ...helpers import batchify, get_attribute, getitem_or_getattr
from ...schema import Family, PythonOnly, serializable
from ...typing import AttributeGetter, Float
from .._nested import NestedMeasure
from ..aggregator import Aggregator, default_aggregator

__all__ = [
    "EDGE_MEASURES",
    "GRAPH_MATCHERS",
    "AnyElement",
    "EdgeSimFunc",
    "ElementMatcher",
    "SameAttribute",
    "SemanticEdge",
]


class ElementMatcher[T](Protocol):
    """Determines whether two graph elements can be legally mapped."""

    def __call__(self, case: T, query: T, /) -> bool: ...


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AnyElement(ElementMatcher[Any]):
    """Allows to map any pair of elements."""

    kind: Literal["any_element"] = field(default="any_element", repr=False)

    def __call__(self, case: Any, query: Any, /) -> bool:
        return True


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SameAttribute(ElementMatcher[Any]):
    """Allows to map two elements only if their values of an attribute are equal, such as their types.

    Args:
        attribute: Name of the attribute.
        value_getter: Reads the attribute of a value, a missing attribute counts as None.
    """

    attribute: str
    value_getter: PythonOnly[AttributeGetter] = getitem_or_getattr
    kind: Literal["same_attribute"] = field(default="same_attribute", repr=False)

    def __call__(self, case: Any, query: Any, /) -> bool:
        return get_attribute(case, self.attribute, self.value_getter) == get_attribute(
            query, self.attribute, self.value_getter
        )


class EdgeSimFunc[E](Protocol):
    """Similarities of edge pairs given the similarities of their endpoints.

    Every item of the batch is a case edge value, a query edge value,
    and the similarities of their `source` and `target` nodes.
    """

    def __call__(
        self, batches: Sequence[tuple[E, E, float, float]], /
    ) -> Sequence[float]: ...


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SemanticEdge[E](EdgeSimFunc[E]):
    """Similarity of two edges, the aggregated similarity of their endpoints scaled by that of the edge values.

    Args:
        value_measure: Similarity of the edge values, 1 for every edge if None.
        aggregator: Aggregates the similarities of the `source` and `target` nodes by these keys.

    Examples:
        >>> from cbrkit.sim import Aggregator
        >>> SemanticEdge()([(None, None, 1.0, 0.5)])
        [0.75]
        >>> SemanticEdge(aggregator=Aggregator(weights={"source": 3}))([(None, None, 1.0, 0.5)])
        [0.875]
    """

    value_measure: NestedMeasure[E, Float] | None = None
    aggregator: Aggregator = default_aggregator
    kind: Literal["semantic_edge"] = field(default="semantic_edge", repr=False)

    def __call__(self, batches: Sequence[tuple[E, E, float, float]], /) -> list[float]:
        endpoints = np.array(
            [(source, target) for _, _, source, target in batches], dtype=np.float64
        ).reshape(-1, 2)
        sims = self.aggregator.columns(
            ("source", "target"), endpoints, np.ones(endpoints.shape, dtype=np.bool_)
        )

        if self.value_measure is not None:
            sims *= np.array(
                [
                    float(sim)
                    for sim in batchify(self.value_measure)(
                        [(x, y) for x, y, _, _ in batches]
                    )
                ],
                dtype=np.float64,
            )

        return sims.tolist()


GRAPH_MATCHERS = Family(
    "element matcher",
    (AnyElement, SameAttribute),
)

EDGE_MEASURES = Family(
    "edge measure",
    (SemanticEdge,),
)
