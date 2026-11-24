from collections.abc import Callable, Mapping
from dataclasses import MISSING, fields
from functools import cache, wraps
from typing import (
    Annotated,
    Any,
    TypeAliasType,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SkipValidation,
    create_model,
    with_config,
)
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema

from ._docstrings import describe

__all__ = [
    "Model",
    "PythonOnly",
    "Unit",
    "serializable",
]

T = TypeVar("T")


class _PythonOnly:
    """Marks the fields of `PythonOnly`."""


# a classic alias, since pydantic ignores field metadata such as `exclude` in `type` aliases
PythonOnly = Annotated[
    SkipValidation[SkipJsonSchema[T]], Field(exclude=True), _PythonOnly()
]
"""A field that only Python code sets, such as a function, left out of serialized forms.

Serializing a member of a `Family` family whose field differs from its default raises,
so a serialized form never silently drops a custom function.
"""


type Unit = Annotated[float, Field(ge=0, le=1)]
"""A number between 0 and 1 such as a similarity, validated in configurations."""


_SERIALIZABLE = ConfigDict(
    extra="forbid", json_schema_extra=describe, arbitrary_types_allowed=True
)
"""The configuration shared by models and `serializable` dataclasses."""


class Model(BaseModel):
    """Base of all models of cbrkit: immutable, strict about unknown fields, and described by their docstrings.

    Their schemas are built on first use, see `extend` for models with further kinds.
    """

    model_config = _SERIALIZABLE | ConfigDict(frozen=True, defer_build=True)


@cache
def _python_only(cls: type[Any]) -> Mapping[str, Any]:
    """Defaults of the `PythonOnly` fields of a dataclass by field name."""
    hints = get_type_hints(cls, include_extras=True)
    defaults: dict[str, Any] = {}

    for f in fields(cls):
        metadata = getattr(hints.get(f.name), "__metadata__", ())

        if any(isinstance(meta, _PythonOnly) for meta in metadata):
            defaults[f.name] = (
                f.default_factory() if f.default_factory is not MISSING else f.default
            )

    return defaults


def _any_annotation(
    annotation: Any, predicate: Callable[[Any], bool], seen: set[int] | None = None
) -> bool:
    """Whether a predicate holds for a type or a type nested in it, also behind type aliases and in metadata."""
    if predicate(annotation):
        return True

    seen = set() if seen is None else seen

    if id(annotation) in seen:
        return False

    seen.add(id(annotation))

    if isinstance(annotation, TypeAliasType):
        return _any_annotation(annotation.__value__, predicate, seen)

    origin = get_origin(annotation)

    return any(
        _any_annotation(arg, predicate, seen)
        for arg in (*get_args(annotation), *([origin] if origin else []))
    )


def _constrained(annotation: Any) -> bool:
    """Whether a type carries a constraint such as `Field(ge=0)`."""
    return any(
        isinstance(meta, FieldInfo) and meta.metadata
        for meta in getattr(annotation, "__metadata__", ())
    )


@cache
def _constraints(cls: type[Any]) -> type[BaseModel] | None:
    """A model of the constrained fields of a dataclass, None if it has none."""
    hints = get_type_hints(cls, include_extras=True)
    constrained: dict[str, Any] = {
        f.name: (hints[f.name], ...)
        for f in fields(cls)
        if f.init and _any_annotation(hints[f.name], _constrained)
    }

    if not constrained:
        return None

    return create_model(
        cls.__name__,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **constrained,
    )


def serializable[T: type[Any]](cls: T) -> T:
    """Lets pydantic validate and serialize a keyword-only dataclass, described by its docstring.

    Unknown fields are rejected, and the JSON Schema takes its descriptions from the Google-style docstring.
    Members of a `Family` family are such classes with a `kind` field.
    Constraints of the field types, such as `Annotated[float, Field(gt=0)]`,
    also hold when Python code constructs the class, raising a `pydantic.ValidationError`.

    Raises:
        TypeError: If the class is no dataclass with keyword-only parameters.

    Examples:
        >>> from dataclasses import dataclass
        >>> @serializable
        ... @dataclass(slots=True, frozen=True, kw_only=True)
        ... class Scale:
        ...     factor: Annotated[float, Field(gt=0)]
        >>> Scale(factor=0)
        Traceback (most recent call last):
        pydantic_core._pydantic_core.ValidationError: 1 validation error for Scale
        ...
    """
    dataclass_fields = getattr(cls, "__dataclass_fields__", None)

    if dataclass_fields is None:
        raise TypeError(f"{cls.__name__} is no dataclass")

    if positional := [
        name for name, f in dataclass_fields.items() if f.init and not f.kw_only
    ]:
        raise TypeError(
            f"{cls.__name__} takes positional parameters {positional}, declare it with kw_only=True"
        )

    init = cls.__init__

    # pydantic validates on its own and builds instances without calling __init__
    @wraps(init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        init(self, *args, **kwargs)

        # the types resolve once the module is loaded, so the first instance decides
        if (model := _constraints(cls)) is None:
            cls.__init__ = init

            return

        model.model_validate({name: getattr(self, name) for name in model.model_fields})

    cls.__init__ = __init__

    return with_config(_SERIALIZABLE)(cls)
