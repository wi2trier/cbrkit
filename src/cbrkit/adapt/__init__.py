"""Adaptation functions for modifying retrieved cases to better suit a query.

This module contains built-in adaptation functions for common data types.
Adaptation functions are used in the reuse phase (`cbrkit.reuse`) to transform
retrieved cases before scoring.
All adaptation functions follow one of three signatures:

- Pair: `adapted = f(case, query)`
- Map: `adapted_casebase = f(casebase, query)`
- Reduce: `key, adapted = f(casebase, query)`

Submodules:
- `cbrkit.adapt.numbers`: Numeric adaptation (e.g., `aggregate` with pooling).
- `cbrkit.adapt.strings`: String adaptation (e.g., `regex` replacement).
- `cbrkit.adapt.generic`: Generic adaptation functions.

Top-Level Functions:
- `attribute_value`: Applies per-attribute adaptation functions to
  attribute-value based cases, analogous to `cbrkit.sim.attribute_value`.
  Supports nesting for object-oriented data structures.

Example:
    >>> adapter = attribute_value(
    ...     attributes={
    ...         "price": numbers.aggregate(pooling="mean"),
    ...         "color": strings.regex("CASE_PATTERN", "QUERY_PATTERN", "REPLACEMENT"),
    ...     }
    ... )
"""

from . import generic, numbers, strings
from .attribute_value import attribute_value

__all__ = [
    "generic",
    "strings",
    "numbers",
    "attribute_value",
]
