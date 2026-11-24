from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, override

from ..helpers import batchify, get_attribute, getitem_or_getattr, replace_attributes
from ..schema import PythonOnly, serializable
from ..typing import AttributeGetter, BatchAdaptationFunc
from ._nested import NestedAdaptation

__all__ = ["AttributeValueAdaptation"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AttributeValueAdaptation[V](BatchAdaptationFunc[V]):
    """Adapt values of attributes using specified adaptation functions.

    This class allows for the adaptation of multiple attributes of a case by applying
    an adaptation function to each attribute, `cbrkit.adapt.generic.Pipe` chains several.
    It supports different data structures like mappings (dictionaries) and dataframes.
    Every adaptation function is called once with the pairs of all cases,
    so a batch adaptation such as `cbrkit.adapt.llm.Rewrite` sends them in one batch.
    An attribute that is missing or `None` in the case or the query is kept as it is.
    The adapted case is a copy, so adaptation cannot write back into the casebase it was retrieved from.

    Args:
        attributes: Adaptation function of every adapted attribute by its name.
        value_getter: Function to retrieve values from objects. Defaults to dictionary/attribute access.
        case_builder: Function that builds the adapted case from the case and the
            adapted attributes. Defaults to a copy that supports mappings,
            dataclasses, and pydantic models, including immutable ones.

    Examples:
        >>> func = AttributeValueAdaptation(attributes={
        ...     "name": lambda x, y: x if x == y else y,
        ...     "age": lambda x, y: x if x > y else y,
        ... })
        >>> func([
        ...     ({"name": "Alice", "age": 30}, {"name": "Peter", "age": 25}),
        ...     ({"name": "Bob"}, {"name": "Bob", "age": 40}),
        ... ])
        [{'name': 'Peter', 'age': 30}, {'name': 'Bob'}]
    """

    attributes: Mapping[str, NestedAdaptation[Any]]
    value_getter: PythonOnly[AttributeGetter] = getitem_or_getattr
    case_builder: PythonOnly[Callable[[Any, Mapping[str, Any]], Any]] = (
        replace_attributes
    )
    kind: Literal["attribute_value"] = field(default="attribute_value", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> Sequence[V]:
        updates: list[dict[str, Any]] = [{} for _ in batches]

        for name, func in self.attributes.items():
            pairs = {
                idx: (case_value, query_value)
                for idx, (case, query) in enumerate(batches)
                if (case_value := get_attribute(case, name, self.value_getter))
                is not None
                and (query_value := get_attribute(query, name, self.value_getter))
                is not None
            }

            if pairs:
                adapted = batchify(func)(list(pairs.values()))

                for idx, value in zip(pairs, adapted, strict=True):
                    updates[idx][name] = value

        return [
            self.case_builder(case, update)
            for (case, _), update in zip(batches, updates, strict=True)
        ]
