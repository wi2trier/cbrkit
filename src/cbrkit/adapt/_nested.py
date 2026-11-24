from typing import Annotated, TypeVar

from ..schema import Family
from ..typing import AnyAdaptationFunc

__all__ = ["NestedAdaptation"]

_V = TypeVar("_V")

# a classic generic alias, since pydantic would name a `type` alias by its type arguments in JSON Schemas
NestedAdaptation = Annotated[AnyAdaptationFunc[_V], Family("adaptation")]
"""An adaptation nested in a member of `cbrkit.adapt.ADAPTATIONS`, which serializes by kind like them,
see `cbrkit.adapt.Adaptation` for fields elsewhere.

Its family is named only, so the modules of the adaptations need not import each other."""
