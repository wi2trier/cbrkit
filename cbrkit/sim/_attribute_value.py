from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, override

import pandas as pd

from cbrkit.helpers import get_metadata, get_name, sim2map
from cbrkit.typing import (
    AggregatorFunc,
    AnnotatedFloat,
    AnySimFunc,
    Float,
    JsonDict,
    SimMap,
    SimMapFunc,
    SupportsMetadata,
)

from ._aggregator import aggregator

__all__ = ["attribute_value", "AttributeValueData", "AttributeValueSim"]

type AttributeValueData = Mapping[Any, Any] | pd.Series[Any]


def _key_getter(obj: AttributeValueData) -> Iterator[str]:
    if isinstance(obj, Mapping):
        yield from obj.keys()
    elif isinstance(obj, pd.Series):
        yield from obj.index
    else:
        raise NotImplementedError()


def _value_getter(obj: AttributeValueData, key: Any) -> Any:
    if isinstance(obj, Mapping | pd.Series):
        return obj[key]
    else:
        return getattr(obj, key)


@dataclass(slots=True, frozen=True)
class AttributeValueSim[S: Float](AnnotatedFloat):
    value: float
    by_attribute: Mapping[str, S]


_aggregator = aggregator()


@dataclass(slots=True, frozen=True)
class attribute_value[K, V, S: Float](
    SimMapFunc[K, V, AttributeValueSim[S]], SupportsMetadata
):
    """
    Similarity function that computes the attribute value similarity between two cases.

    Args:
        attributes: A mapping of attribute names to the similarity functions to be used for those attributes. Takes precedence over types.
        types: A mapping of attribute types to the similarity functions to be used for those types.
        types_fallback: A similarity function to be used as a fallback when no specific similarity function
            is defined for an attribute type.
        aggregator: A function that aggregates the local similarity scores for each attribute into a single global similarity.
        value_getter: A function that retrieves the value of an attribute from a case.
        key_getter: A function that retrieves the attribute names from a target case.

    Examples:
        >>> equality = lambda x, y: 1.0 if x == y else 0.0
        >>> sim = attribute_value(
        ...     attributes={
        ...         "name": equality,
        ...         "age": equality,
        ...     },
        ... )
        >>> scores = sim(
        ...     {
        ...         "a": {"name": "John", "age": 25},
        ...         "b": {"name": "Jane", "age": 30},
        ...     },
        ...     {"name": "John", "age": 30},
        ... )
        >>> scores["a"]
        AttributeValueSim(value=0.5, by_attribute={'age': 0.0, 'name': 1.0})
        >>> scores["b"]
        AttributeValueSim(value=0.5, by_attribute={'age': 1.0, 'name': 0.0})
    """

    attributes: Mapping[str, AnySimFunc[K, Any, S]] = field(default_factory=dict)
    types: Mapping[type[Any], AnySimFunc[K, Any, S]] = field(default_factory=dict)
    types_fallback: AnySimFunc[K, Any, S] | None = None
    aggregator: AggregatorFunc[str, S] = _aggregator
    value_getter: Callable[[Any, str], Any] = _value_getter
    key_getter: Callable[[Any], Iterator[str]] = _key_getter

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "attributes": {
                key: get_metadata(value) for key, value in self.attributes.items()
            },
            "types": {
                key.__name__: get_metadata(value) for key, value in self.types.items()
            },
            "types_fallback": get_metadata(self.types_fallback),
            "aggregator": get_metadata(self.aggregator),
            "value_getter": get_name(self.value_getter),
            "key_getter": get_name(self.key_getter),
        }

    @override
    def __call__(self, x_map: Mapping[K, V], y: V) -> SimMap[K, AttributeValueSim[S]]:
        local_sims: defaultdict[K, dict[str, S]] = defaultdict(dict)

        attribute_names = (
            set(self.attributes).intersection(self.key_getter(y))
            if len(self.attributes) > 0
            and len(self.types) == 0
            and self.types_fallback is None
            else set(self.key_getter(y))
        )

        for attr_name in attribute_names:
            x_attributes = {
                key: self.value_getter(value, attr_name) for key, value in x_map.items()
            }
            y_attribute = self.value_getter(y, attr_name)
            attr_type = type(y_attribute)

            sim_func = (
                self.attributes[attr_name]
                if attr_name in self.attributes
                else self.types.get(attr_type, self.types_fallback)
            )

            assert (
                sim_func is not None
            ), f"no similarity function for {attr_name} with type {attr_type}"

            sim_func = sim2map(sim_func)
            sim_func_result = sim_func(x_attributes, y_attribute)

            for key, sim in sim_func_result.items():
                local_sims[key][attr_name] = sim

        return {
            key: AttributeValueSim(self.aggregator(sims), sims)
            for key, sims in local_sims.items()
        }
