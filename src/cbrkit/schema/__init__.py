"""Serialized forms and JSON Schemas derived from classes and their docstrings.

Descriptions come from Google-style docstrings,
the summary of a class or function and the `Args` section for its parameters,
so the code stays the only source of the schemas offered to users and language models.

A family of classes (`Family`) serializes each member with a `kind` discriminator.
Its members are frozen, keyword-only dataclasses with a `kind` field,
and annotating a field with the family lets pydantic validate and dump any member.

Configurable fields follow one rule, so every field has a single JSON Schema shape:
- Behaviour is an object of a kind of a family, such as a measure, a pooling function or a vector metric,
  also when it has no parameters (`Mean()`), and its default is an instance, never None or a string.
- A mode is a `Literal`, a closed and parameterless switch that the owning class interprets itself.
  A two-valued switch is a `bool`, and parameters that only one value needs make that value an object.
- A reference is a `str` that names something outside the configuration,
  such as a model, a `HostFunction` of the host, a table, a column or a path.
- A field has one form and a concept one vocabulary, so a field never accepts a union of these.
None stays allowed where it switches a feature off, such as a missing duplicate threshold.
"""

from ._docstrings import class_docs, describe, parse_docstring
from ._family import Family, extend, resolve, using
from ._host import HostFunction, HostRef
from ._models import Model, PythonOnly, Unit, serializable

__all__ = [
    "Family",
    "HostFunction",
    "HostRef",
    "Model",
    "PythonOnly",
    "Unit",
    "class_docs",
    "describe",
    "extend",
    "parse_docstring",
    "resolve",
    "serializable",
    "using",
]
