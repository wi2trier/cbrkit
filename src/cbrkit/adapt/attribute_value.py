from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, override

from ..helpers import produce_sequence
from ..typing import AdaptationFunc, MaybeSequence

__all__ = ["attribute_value"]


def default_value_getter(obj: Any, key: Any) -> Any:
    if hasattr(obj, "__getitem__"):
        return obj[key]
    else:
        return getattr(obj, key)


def default_value_setter(obj: Any, key: Any, value: Any) -> None:
    if hasattr(obj, "__setitem__"):
        obj[key] = value
    else:
        setattr(obj, key, value)


@dataclass(slots=True, frozen=True)
class attribute_value[V](AdaptationFunc[V]):
    """Adapt values of attributes using specified adaptation functions.

    This class allows for the adaptation of multiple attributes of a case by applying
    one or more adaptation functions to each attribute. It supports different data structures
    like mappings (dictionaries) and dataframes

    Args:
        attributes: A mapping of attribute names to either single adaptation functions or
            sequences of adaptation functions that will be applied in order.
        value_getter: Function to retrieve values from objects. Defaults to dictionary/attribute access.
        value_setter: Function to set values on objects. Defaults to dictionary/attribute assignment.

    Returns:
        A new case with adapted attribute values.

    Examples:
        >>> func = attribute_value({
        ...     "name": lambda x, y: x if x == y else y,
        ...     "age": lambda x, y: x if x > y else y,
        ... })
        >>> result = func(
        ...     {"name": "Alice", "age": 30},
        ...     {"name": "Peter", "age": 25}
        ... )
        >>> result
        {'name': 'Peter', 'age': 30}
    """

    attributes: Mapping[str, MaybeSequence[AdaptationFunc[Any]]]
    value_getter: Callable[[Any, str], Any] = default_value_getter
    value_setter: Callable[[Any, str, Any], None] = default_value_setter

    @override
    def __call__(self, case: V, query: V) -> V:
        for attr_name in self.attributes:
            adapt_funcs = produce_sequence(self.attributes[attr_name])

            case_attr_value = self.value_getter(case, attr_name)
            query_attr_value = self.value_getter(query, attr_name)

            for adapt_func in adapt_funcs:
                case_attr_value = adapt_func(case_attr_value, query_attr_value)

            self.value_setter(case, attr_name, case_attr_value)

        return case
