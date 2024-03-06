from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Generic

import pandas as pd

from cbrkit.helpers import sim2map
from cbrkit.typing import (
    AggregatorFunc,
    AnySimFunc,
    Casebase,
    FloatProtocol,
    KeyType,
    SimMap,
    SimMapFunc,
    SimType,
    ValueType,
)

from ._aggregator import aggregator

__all__ = ["attribute_value", "AttributeValueData", "AttributeValueSim"]

AttributeValueData = Mapping[Any, Any] | pd.Series


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
class AttributeValueSim(FloatProtocol, Generic[SimType]):
    value: float
    by_attribute: Mapping[str, SimType]


_aggregator = aggregator()


def attribute_value(
    attributes: Mapping[str, AnySimFunc[KeyType, Any, SimType]] | None = None,
    types: Mapping[type[Any], AnySimFunc[KeyType, Any, SimType]] | None = None,
    types_fallback: AnySimFunc[KeyType, Any, SimType] | None = None,
    aggregator: AggregatorFunc[str, SimType] = _aggregator,
    value_getter: Callable[[Any, str], Any] = _value_getter,
    key_getter: Callable[[Any], Iterator[str]] = _key_getter,
) -> SimMapFunc[Any, AttributeValueData, AttributeValueSim[SimType]]:
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

    attributes_map: Mapping[str, AnySimFunc[KeyType, Any, SimType]] = (
        {} if attributes is None else attributes
    )
    types_map: Mapping[type[Any], AnySimFunc[KeyType, Any, SimType]] = (
        {} if types is None else types
    )

    def wrapped_func(
        x_map: Casebase[KeyType, ValueType], y: ValueType
    ) -> SimMap[KeyType, AttributeValueSim[SimType]]:
        local_sims: defaultdict[KeyType, dict[str, SimType]] = defaultdict(dict)

        attribute_names = (
            set(attributes_map).intersection(key_getter(y))
            if len(attributes_map) > 0
            and len(types_map) == 0
            and types_fallback is None
            else set(key_getter(y))
        )

        for attr_name in attribute_names:
            x_attributes = {
                key: value_getter(value, attr_name) for key, value in x_map.items()
            }
            y_attribute = value_getter(y, attr_name)
            attr_type = type(y_attribute)

            sim_func = (
                attributes_map[attr_name]
                if attr_name in attributes_map
                else types_map.get(attr_type, types_fallback)
            )

            assert (
                sim_func is not None
            ), f"no similarity function for {attr_name} with type {attr_type}"

            sim_func = sim2map(sim_func)
            sim_func_result = sim_func(x_attributes, y_attribute)

            for key, sim in sim_func_result.items():
                local_sims[key][attr_name] = sim

        return {
            key: AttributeValueSim(aggregator(sims), sims)
            for key, sims in local_sims.items()
        }

    return wrapped_func
