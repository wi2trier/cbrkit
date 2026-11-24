from typing import Annotated, TypeVar

from ..schema import Family
from ..typing import AnySimFunc, Float

__all__ = ["NestedMeasure"]

_V = TypeVar("_V")
_S = TypeVar("_S", bound=Float, default=float)

# a classic generic alias, since pydantic would name a `type` alias by its type arguments in JSON Schemas
NestedMeasure = Annotated[AnySimFunc[_V, _S], Family("measure")]
"""A measure nested in a member of `cbrkit.sim.MEASURES`, which serializes by kind like them,
see `cbrkit.sim.Measure` for fields elsewhere.

Its family is named only, so the modules of the measures need not import each other."""
