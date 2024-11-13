from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import pandas as pd

from cbrkit.helpers import get_metadata
from cbrkit.typing import (
    AdaptPairFunc,
    JsonDict,
    SupportsMetadata,
)

__all__ = ["attribute_value", "AttributeValueData"]

# TODO: Add Polars
type AttributeValueData = Mapping | pd.Series


def default_value_getter(obj: AttributeValueData, key: Any) -> Any:
    if isinstance(obj, Mapping | pd.Series):
        return obj[key]
    else:
        return getattr(obj, key)


def default_value_setter(obj: AttributeValueData, key: Any, value: Any) -> None:
    if isinstance(obj, MutableMapping):
        obj[key] = value
    else:
        setattr(obj, key, value)


@dataclass(slots=True, frozen=True)
class attribute_value[V](AdaptPairFunc[V], SupportsMetadata):
    attributes: Mapping[str, AdaptPairFunc[Any] | Sequence[AdaptPairFunc[Any]]]
    value_getter: Callable[[Any, str], Any] = default_value_getter
    value_setter: Callable[[Any, str, Any], None] = default_value_setter

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "attributes": {
                key: get_metadata(value) for key, value in self.attributes.items()
            },
            "value_getter": get_metadata(self.value_getter),
            "value_setter": get_metadata(self.value_setter),
        }

    @override
    def __call__(self, case: V, query: V) -> V:
        for attr_name in self.attributes:
            adapt_funcs = self.attributes[attr_name]

            if not isinstance(adapt_funcs, Sequence):
                adapt_funcs = [adapt_funcs]

            case_attr_value = self.value_getter(case, attr_name)
            query_attr_value = self.value_getter(query, attr_name)

            for adapt_func in adapt_funcs:
                case_attr_value = adapt_func(case_attr_value, query_attr_value)

            self.value_setter(case, attr_name, case_attr_value)

        return case
