"""Pydantic models of the cases and queries of a specification, derived from its measures."""

import inspect
import types
from collections.abc import Iterable, Mapping
from typing import Annotated, Any, TypeVar, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    WithJsonSchema,
    create_model,
)
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError

from ..model.graph import Graph
from ..schema import parse_docstring
from ..sim.attribute_value import AttributeValue
from ..sim.generic import Equality
from ..sim.numbers import Number
from ..typing import BatchSimFunc, Casebase, ColumnSimFunc, SimFunc
from ._config import Config

__all__ = ["case_model"]

MAX_VALUES = 50
"""Most values of a text attribute compared for equality that a query schema lists."""


def _concrete(annotation: Any) -> Any:
    """Replaces type variables and types without a JSON Schema by `Any`."""
    if isinstance(annotation, TypeVar):
        return Any

    if get_origin(annotation) is Graph:
        return Graph

    if args := get_args(annotation):
        origin = get_origin(annotation)

        if origin in (types.UnionType, None):
            return annotation

        try:
            annotation = origin[tuple(_concrete(arg) for arg in args)]
        except TypeError:
            return Any

    try:
        TypeAdapter(annotation).json_schema()
    except PydanticSchemaGenerationError, PydanticInvalidForJsonSchema:
        return Any

    return annotation


def _value_type(measure: object, *, stored: bool = False) -> Any:
    """The type of the values that a measure compares, read from its class.

    Attribute-value measures give a model of their attributes, which allows unmeasured fields of `stored` cases,
    graph measures the graph of `cbrkit.model.graph`, validated from and serialized to its JSON form.

    Examples:
        >>> from cbrkit.sim.numbers import Linear
        >>> _value_type(Linear(max_distance=1))
        Number
    """
    if isinstance(measure, AttributeValue):
        return _model("Attributes", dict(measure.attributes), stored=stored)

    for cls in type(measure).__mro__:
        for base in (
            types.get_original_bases(cls) if "__orig_bases__" in vars(cls) else ()
        ):
            if get_origin(base) in (SimFunc, BatchSimFunc, ColumnSimFunc):
                return _concrete(get_args(base)[0])

    return Any


def _description(measure: object) -> str:
    summary = parse_docstring(inspect.getdoc(type(measure)) or "")[0].split("\n")[0]

    return f"Compared by {type(measure).__name__}: {summary}"


def _observed(
    measure: object, value_type: Any, values: Iterable[Any]
) -> tuple[Any, dict[str, Any]]:
    """The type replacing *value_type*, None to keep it, and the schema keywords of an attribute as its *values* show it.

    Text compared for equality lists at most `MAX_VALUES` values as an enum, which the schema shows without validating,
    since a stored case may add one, and a number gives its smallest and largest value as examples.
    Other measures leave the values unread.

    Examples:
        >>> _observed(Equality(), Any, ["b", None, "a", "b"])[0].__metadata__[0].json_schema
        {'type': 'string', 'enum': ['a', 'b']}
        >>> _observed(None, Number, [3, 1.5, None])
        (None, {'examples': [1.5, 3]})
    """
    if isinstance(measure, Equality):
        distinct: set[str] = set()

        for value in values:
            if value is None or value in distinct:
                continue

            if not isinstance(value, str) or len(distinct) == MAX_VALUES:
                return None, {}

            distinct.add(value)

        enum = WithJsonSchema({"type": "string", "enum": sorted(distinct)})

        return (Annotated[str, enum] if distinct else None), {}

    if value_type is not Number:
        return None, {}

    numbers = [value for value in values if value is not None]

    if numbers and all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in numbers
    ):
        return None, {"examples": [min(numbers), max(numbers)]}

    return None, {}


def _model(
    name: str,
    measures: Mapping[str, object],
    *,
    stored: bool,
    cases: Casebase[Any, Any] | None = None,
) -> type[BaseModel]:
    fields: dict[str, Any] = {}

    for attribute, measure in measures.items():
        value_type = _value_type(measure, stored=stored)
        extra: dict[str, Any] = {}

        if cases is not None:
            observed, extra = _observed(
                measure,
                value_type,
                (
                    case.get(attribute)
                    for case in cases.values()
                    if isinstance(case, Mapping)
                ),
            )
            value_type = observed or value_type

        fields[attribute] = (
            value_type | None,
            Field(
                default=None,
                description=_description(measure),
                json_schema_extra=extra or None,
            ),
        )

    return create_model(
        name, __config__=ConfigDict(extra="allow" if stored else "forbid"), **fields
    )


def case_model(
    config: Config, stored: bool = False, cases: Casebase[Any, Any] | None = None
) -> type[BaseModel]:
    """The model of the cases and queries of a configuration, every attribute being optional.

    Queries may leave out attributes, so `model_dump(exclude_unset=True)` gives the partial query.
    Unknown attributes are rejected, except in the model of stored cases, which may have more columns
    and is named `StoredCase` instead of `Case`.

    Args:
        config: The configuration whose similarity defines the attributes.
        stored: Whether the model is that of stored cases, which allows attributes without a measure.
        cases: The case base whose records the schema of the queries describes, listing the values
            of a text attribute compared for equality and the range of a number.

    Examples:
        >>> from cbrkit.io import Source
        >>> from cbrkit.spec import infer
        >>> config = infer({0: {"price": 1.0, "fuel": "petrol"}}, Source(format="json"))
        >>> sorted(case_model(config).model_json_schema()["properties"])
        ['fuel', 'price']
        >>> cases = dict(enumerate(
        ...     {"price": price, "fuel": fuel} for price, fuel in [(1.0, "petrol"), (2.0, "diesel")] * 2
        ... ))
        >>> config = infer(cases, Source(format="json"))
        >>> schema = case_model(config, cases=cases).model_json_schema()["properties"]
        >>> schema["fuel"]["anyOf"][0], schema["price"]["examples"]
        ({'enum': ['diesel', 'petrol'], 'type': 'string'}, [1.0, 2.0])
    """
    measure = config.measure
    name = "StoredCase" if stored else "Case"

    if isinstance(measure, AttributeValue):
        return _model(
            name,
            dict(measure.attributes),
            stored=stored,
            cases=None if stored else cases,
        )

    return create_model(
        name, __base__=RootModel, root=(_value_type(measure, stored=stored), ...)
    )
