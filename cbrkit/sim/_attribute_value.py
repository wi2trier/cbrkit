from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import pandas as pd

from cbrkit.helpers import SimSeqWrapper, get_metadata
from cbrkit.typing import (
    AggregatorFunc,
    AnnotatedFloat,
    AnySimFunc,
    Float,
    JsonDict,
    SimSeq,
    SimSeqFunc,
    SupportsMetadata,
)

from ._aggregator import aggregator

__all__ = ["attribute_value", "AttributeValueData", "AttributeValueSim"]

# TODO: Add Polars
type AttributeValueData = Mapping | pd.Series


def default_value_getter(obj: AttributeValueData, key: Any) -> Any:
    if isinstance(obj, Mapping | pd.Series):
        return obj[key]
    else:
        return getattr(obj, key)


@dataclass(slots=True, frozen=True)
class AttributeValueSim[S: Float](AnnotatedFloat):
    value: float
    attributes: Mapping[str, S]


default_aggregator = aggregator()


@dataclass(slots=True, frozen=True)
class attribute_value[K, V, S: Float](
    SimSeqFunc[V, AttributeValueSim[S]], SupportsMetadata
):
    """
    Similarity function that computes the attribute value similarity between two cases.

    Args:
        attributes: A mapping of attribute names to the similarity functions to be used for those attributes.
        aggregator: A function that aggregates the local similarity scores for each attribute into a single global similarity.
        value_getter: A function that retrieves the value of an attribute from a case.

    Examples:
        >>> equality = lambda x, y: 1.0 if x == y else 0.0
        >>> sim = attribute_value(
        ...     attributes={
        ...         "name": equality,
        ...         "age": equality,
        ...     },
        ... )
        >>> scores = sim(
        ...     [
        ...         ({"name": "John", "age": 25}, {"name": "John", "age": 30}),
        ...         ({"name": "Jane", "age": 30}, {"name": "John", "age": 30}),
        ...     ]
        ... )
        >>> scores[0]
        AttributeValueSim(value=0.5, attributes={'name': 1.0, 'age': 0.0})
        >>> scores[1]
        AttributeValueSim(value=0.5, attributes={'name': 0.0, 'age': 1.0})
    """

    attributes: Mapping[str, AnySimFunc[Any, S]]
    aggregator: AggregatorFunc[str, S] = default_aggregator
    value_getter: Callable[[Any, str], Any] = default_value_getter

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "attributes": {
                key: get_metadata(value) for key, value in self.attributes.items()
            },
            "aggregator": get_metadata(self.aggregator),
            "value_getter": get_metadata(self.value_getter),
        }

    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> SimSeq[AttributeValueSim[S]]:
        if len(pairs) == 0:
            return []

        local_sims: list[dict[str, S]] = [dict() for _ in range(len(pairs))]

        for attr_name in self.attributes:
            attribute_values = [
                (self.value_getter(x, attr_name), self.value_getter(y, attr_name))
                for x, y in pairs
            ]
            sim_func = SimSeqWrapper(self.attributes[attr_name])
            sim_func_result = sim_func(attribute_values)

            for idx, sim in enumerate(sim_func_result):
                local_sims[idx][attr_name] = sim

        return [AttributeValueSim(self.aggregator(sims), sims) for sims in local_sims]
