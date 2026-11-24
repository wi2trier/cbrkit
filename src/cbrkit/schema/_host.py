from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypeVar, get_origin

from pydantic import GetCoreSchemaHandler, ValidationInfo
from pydantic_core import CoreSchema, core_schema

from ..errors import Unbound

__all__ = [
    "HostFunction",
    "HostRef",
]

H = TypeVar("H")


@dataclass(slots=True, frozen=True)
class HostFunction[F]:
    """A function of the host that serialized forms reference by its name only.

    Validating a name binds the function of the same name from the mapping that the pydantic validation context
    holds under the class, such as `{Embedder: embedders}`, which `cbrkit.spec.bind` provides.
    With such a mapping, every name of the class has to be bound, otherwise validation raises `cbrkit.errors.Unbound`.
    Subclasses such as `cbrkit.sim.embed.Embedder` and `cbrkit.sim.llm.LanguageModel` add how the function is called.

    Fields are annotated with the bare subclass, such as `LanguageModel`,
    or with `HostRef` around a parametrized one, such as `HostRef[Embedder[E]]`,
    since pydantic validates a parametrized dataclass by its fields and skips the binding.

    Args:
        name: Name of the function, the key of the providers section.
        func: The bound function, None until the host binds it.

    Examples:
        >>> from pydantic import TypeAdapter
        >>> adapter = TypeAdapter(HostFunction)
        >>> adapter.validate_python("f", context={HostFunction: {"f": print}}).func is print
        True
        >>> adapter.validate_python("g", context={HostFunction: {}})
        Traceback (most recent call last):
        cbrkit.errors.Unbound: HostFunction 'g' is not bound to a function
    """

    name: str
    func: F | None = field(default=None, compare=False)

    def bound(self) -> F:
        """The bound function.

        Raises:
            cbrkit.errors.Unbound: If no function is bound to the name.
        """
        if self.func is None:
            raise Unbound(
                f"{type(self).__name__} {self.name!r} is not bound to a function"
            )

        return self.func

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return _binding_schema(cls)


def _binding_schema[F](cls: type[HostFunction[F]]) -> CoreSchema:
    """Validates a name by binding the function of the same name, and serializes the name only."""

    def bind(value: HostFunction[F] | str, info: ValidationInfo) -> HostFunction[F]:
        ref = cls(value) if isinstance(value, str) else value
        funcs: Mapping[str, F] | None = (info.context or {}).get(cls)

        if funcs is None or ref.func is not None:
            return ref

        if ref.name not in funcs:
            raise Unbound(f"{cls.__name__} {ref.name!r} is not bound to a function")

        return replace(ref, func=funcs[ref.name])

    name = core_schema.str_schema(min_length=1)

    return core_schema.json_or_python_schema(
        json_schema=core_schema.with_info_after_validator_function(bind, name),
        python_schema=core_schema.with_info_after_validator_function(
            bind,
            core_schema.union_schema([core_schema.is_instance_schema(cls), name]),
        ),
        serialization=core_schema.plain_serializer_function_ser_schema(
            lambda ref: ref.name
        ),
        ref=cls.__name__,
    )


class _Binding:
    """Marks the fields of `HostRef`."""

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return _binding_schema(get_origin(source) or source)


# a classic alias, since pydantic names a definition after a `type` alias in JSON Schemas
HostRef = Annotated[H, _Binding()]
"""A field holding a parametrized `HostFunction`, such as `HostRef[Embedder[E]]`, which binds it like the bare class."""
