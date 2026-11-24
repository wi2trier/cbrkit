"""A `Family` is an immutable set of serializable dataclasses that share a `kind` tag.
It validates and serializes its members by that tag and builds the JSON Schema offering all of them."""

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, get_type_hints

from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    SerializerFunctionWrapHandler,
    create_model,
)
from pydantic_core import CoreSchema, core_schema

from ._models import _any_annotation, _python_only

__all__ = [
    "Family",
    "extend",
    "resolve",
    "using",
]


def _mentions(annotation: Any, target: Family | type[Any]) -> bool:
    return annotation is target or (
        isinstance(annotation, Family)
        and isinstance(target, Family)
        and annotation.name == target.name
    )


def _unchanged[T](value: T) -> T:
    return value


def nests(cls: type[Any], target: Family | type[Any]) -> bool:
    """Whether a field of a dataclass holds members of a family or instances of a class."""
    hints = get_type_hints(cls, include_extras=True)

    return any(
        _any_annotation(hints[name], lambda annotation: _mentions(annotation, target))
        for name in getattr(cls, "__dataclass_fields__", {})
        if name in hints
    )


def is_member(cls: type[Any]) -> bool:
    """Whether a class is a `serializable` dataclass with a kind field, which a `Family` takes as a member.

    Only the final class of a slotted dataclass is `serializable`, not the class it replaced.
    """
    kind = getattr(
        getattr(cls, "__dataclass_fields__", {}).get("kind"), "default", None
    )

    return isinstance(kind, str) and "__pydantic_config__" in vars(cls)


def _kind(cls: type[Any]) -> str:
    if not is_member(cls):
        raise TypeError(
            f"{cls.__name__} is no serializable dataclass with a kind field"
        )

    return cls.__dataclass_fields__["kind"].default


@dataclass(slots=True, frozen=True)
class Family:
    """An immutable family of classes that serialize with a `kind` discriminator.

    Members are `serializable` dataclasses with a field like
    `kind: Literal["linear"] = field(default="linear", repr=False)`.
    A field annotated with a family, as in `Annotated[PoolingFunc[float], POOLINGS]`,
    validates and serializes every member by its kind,
    and its JSON Schema offers all members with descriptions from their docstrings.
    JSON Schemas list the members once, as a definition named by the `title` of the family,
    and name each member by its class, so the classes of all families need distinct names.

    A family is a value: `|` combines two families of the same name into a new one,
    `using` adds families to the pydantic schemas built in its block,
    and `extend` builds a model whose fields accept the members of further families.
    A family whose members nest it across modules, such as the measures, is declared by its name alone, as in `Family("measure")`,
    so that the modules of the members need not import each other.
    Inside a member, such a field means the family that is being built, so the members nest each other without importing it.

    Args:
        name: Name of the family in words, for error messages and JSON Schemas.
        members: The member classes.

    Raises:
        TypeError: If a member is no serializable dataclass with a kind field.
        ValueError: If two members have the same kind.

    Examples:
        >>> from cbrkit.sim.pooling import Max, Mean
        >>> family = Family("pooling", (Mean,)) | Family("pooling", (Max,))
        >>> sorted(family.by_kind)
        ['max', 'mean']
    """

    name: str
    members: tuple[type[Any], ...] = ()
    by_kind: Mapping[str, type[Any]] = field(init=False, repr=False, compare=False)
    """The members by their kind."""

    def __post_init__(self) -> None:
        kinds = [_kind(cls) for cls in self.members]

        if duplicates := sorted({kind for kind in kinds if kinds.count(kind) > 1}):
            raise ValueError(f"Duplicate {self.name} kinds {duplicates}")

        object.__setattr__(self, "by_kind", dict(zip(kinds, self.members, strict=True)))

    @property
    def title(self) -> str:
        """Name of the family in JSON Schemas, such as `Measure`."""
        return "".join(word[:1].upper() + word[1:] for word in self.name.split())

    def __or__(self, other: Family) -> Family:
        """The members of both families, which need the same name.

        Raises:
            ValueError: If the families have different names.
        """
        if other.name != self.name:
            raise ValueError(f"Cannot combine {self.name} and {other.name} families")

        return Family(self.name, tuple(dict.fromkeys((*self.members, *other.members))))

    def leaves(self) -> Family:
        """The members that nest no other members, for example leaf similarity measures."""
        return Family(
            f"leaf {self.name}",
            tuple(cls for cls in self.members if not nests(cls, self)),
        )

    def excluding(self, held: type[Any]) -> Family:
        """The members none of whose fields holds instances of a class.

        Args:
            held: The class, for example `cbrkit.sim.llm.LanguageModel` for the members without a language model.
        """
        return Family(
            f"{self.name} without {held.__name__}",
            tuple(cls for cls in self.members if not nests(cls, held)),
        )

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        family = resolve(self)

        if not family.members:
            raise TypeError(
                f"No {family.name} members are given, it only stands for the family of a member that nests it"
            )

        # a field declared by the name of the family inside a member means the family itself
        with using(family):
            if len(family.members) == 1:
                schema = handler.generate_schema(family.members[0])
            else:
                # a definition named by the family, so that its members are listed once
                schema = core_schema.tagged_union_schema(
                    {
                        kind: handler.generate_schema(cls)
                        for kind, cls in family.by_kind.items()
                    },
                    discriminator="kind",
                    ref=f"{family.title}:{hash(family.members)}",
                )

        return core_schema.no_info_after_validator_function(
            _unchanged,
            schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                family._serialize, schema=schema
            ),
        )

    def _serialize(self, value: Any, handler: SerializerFunctionWrapHandler) -> Any:
        """Serializes a member, refusing values that would not round-trip.

        Raises:
            ValueError: If the value is no member,
                or one of its `PythonOnly` fields differs from its default.
        """
        cls = type(value)

        if cls not in self.members:
            raise ValueError(
                f"{cls.__name__} is no {self.name} kind and cannot be serialized"
            )

        for name, default in _python_only(cls).items():
            if getattr(value, name) != default:
                raise ValueError(
                    f"{cls.__name__}.{name} holds a custom value that cannot be serialized"
                )

        return handler(value)


_SCOPE: ContextVar[Mapping[str, Family]] = ContextVar(
    "families", default=MappingProxyType({})
)


def resolve(family: Family) -> Family:
    """The members that a field annotated with a family accepts.

    These are the members of the family and those of the families of the same name that `using` adds.

    Examples:
        >>> from cbrkit.sim import MEASURES
        >>> "linear" in resolve(MEASURES).by_kind
        True
    """
    if (added := _SCOPE.get().get(family.name)) is not None:
        return family | added

    return family


@contextmanager
def using(*families: Family) -> Generator[None]:
    """Adds the members of families to the fields annotated with families of the same names.

    Applies to the pydantic schemas built inside the block, see `extend` for models.

    Args:
        families: The families whose members are added.
    """
    scope = dict(_SCOPE.get())

    for family in families:
        scope[family.name] = (
            scope[family.name] | family if family.name in scope else family
        )

    token = _SCOPE.set(scope)

    try:
        yield
    finally:
        _SCOPE.reset(token)


def extend[M: BaseModel](model: type[M], *families: Family) -> type[M]:
    """A subclass of a model whose fields also accept the members of further families.

    Nested fields accept them too, such as the attributes of an `AttributeValue` measure,
    so a custom measure serializes wherever a built-in one does.

    Args:
        model: The model, such as `cbrkit.spec.Config`.
        families: The families whose members are added.

    Examples:
        >>> from dataclasses import dataclass, field
        >>> from typing import Literal
        >>> from cbrkit.schema import serializable
        >>> from cbrkit.spec import Config
        >>> @serializable
        ... @dataclass(slots=True, frozen=True, kw_only=True)
        ... class Prefix:
        ...     '''Whether the case starts with the query.'''
        ...     kind: Literal["prefix"] = field(default="prefix", repr=False)
        ...     def __call__(self, x: str, y: str) -> float:
        ...         return float(x.startswith(y))
        >>> Extended = extend(Config, Family("measure", (Prefix,)))
        >>> Extended.model_validate(
        ...     {"source": {"format": "json"}, "fingerprint": "", "measure": {"kind": "prefix"}}
        ... ).measure
        Prefix()
    """
    built = create_model(model.__name__, __base__=model, __module__=model.__module__)

    with using(*families):
        built.model_rebuild(force=True)

    return built
