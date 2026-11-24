"""Adaptation functions for modifying retrieved cases to better suit a query.

This module contains built-in adaptation functions for common data types.
Adaptation functions are used in the reuse phase (`cbrkit.reuse`) to transform
retrieved cases before scoring, and they take a case and a query: `adapted = f(case, query)`.
The built-in functions are the family `ADAPTATIONS`,
so they serialize by their `kind` like the similarity measures, for instance in `cbrkit.spec`.

Submodules:
- `cbrkit.adapt.numbers`: Numeric adaptation (e.g., `Aggregate` with pooling).
- `cbrkit.adapt.strings`: String adaptation (e.g., `RegexAdaptation` replacement).
- `cbrkit.adapt.generic`: Generic adaptation functions (`Null`, `Union`, and `Pipe` for chaining).

Top-Level Adaptations:
- `AttributeValueAdaptation`: Applies per-attribute adaptation functions to
  attribute-value based cases, analogous to `cbrkit.sim.AttributeValue`.
  Supports nesting for object-oriented data structures.

Example:
    >>> adapter = AttributeValueAdaptation(
    ...     attributes={
    ...         "price": numbers.Aggregate(),
    ...         "color": strings.RegexAdaptation(case_pattern="CASE_PATTERN", query_pattern="QUERY_PATTERN", replacement="REPLACEMENT"),
    ...     }
    ... )
"""

from typing import Annotated, TypeVar

from ..schema import Family
from ..typing import AnyAdaptationFunc
from . import generic, numbers, strings
from .attribute_value import AttributeValueAdaptation

ADAPTATIONS = Family(
    "adaptation",
    (
        AttributeValueAdaptation,
        generic.Null,
        generic.Pipe,
        generic.Union,
        numbers.Aggregate,
        strings.RegexAdaptation,
    ),
)
"""The built-in adaptations, which serialize by their kind, for instance in `cbrkit.spec`."""

# classic generic aliases, since pydantic would name a `type` alias by its type arguments in JSON Schemas
_V = TypeVar("_V")

Adaptation = Annotated[AnyAdaptationFunc[_V], ADAPTATIONS]
"""An adaptation or batch adaptation function that serializes by kind as one of `ADAPTATIONS`, for the fields of classes above `cbrkit.adapt`."""

__all__ = [
    "ADAPTATIONS",
    "Adaptation",
    "AttributeValueAdaptation",
    "generic",
    "numbers",
    "strings",
]
