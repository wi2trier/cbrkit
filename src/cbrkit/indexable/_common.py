"""Shared helpers of the indexable storage backends."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Protocol, cast

from pydantic import BaseModel

from ..sim.embed.metrics import Cosine, Dot, Euclidean


class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


type _DataclassOrModel = type[_DataclassInstance | BaseModel]
"""A dataclass or pydantic model class, whose fields are the columns of a storage."""


def model_columns(model: _DataclassOrModel) -> tuple[str, ...]:
    """Return the declared field names of a dataclass or pydantic model."""
    if is_dataclass(model):
        return tuple(f.name for f in fields(model))
    return tuple(cast(type[BaseModel], model).model_fields)


def make_codec[V](
    model: type[V] | None,
    value_column: str,
    *,
    key_column: str | None = None,
) -> RowCodec[V]:
    """Build a `RowCodec` for a storage backend.

    The casebase key is never part of the codec payload, this is the
    single, uniform policy across backends.  How the key is kept out
    depends on where the backend stores it:

    * **column-oriented** backends (e.g. `lancedb`) keep the key in a
      dedicated `key_column` that shares the column namespace with the
      model's fields, so it is excluded from the payload here.
    * **record-oriented** backends (e.g. `zvec`) store the key
      as the native record id, a separate namespace, and pass
      `key_column=None` since no model column can collide with it.

    Without a model the value is plain text stored under `value_column`.
    The resulting `codec.columns` is the authoritative list of
    cbrkit-owned columns, so backends read it directly instead of
    recomputing it.
    """
    if model is None:
        return RowCodec(columns=(value_column,), value_column=value_column)

    names = model_columns(cast(_DataclassOrModel, model))

    return RowCodec(
        model=model,
        columns=tuple(c for c in names if c != key_column),
        value_column=value_column,
        key_column=key_column,
    )


@dataclass(slots=True, frozen=True)
class RowCodec[V]:
    """Convert a casebase value `V` to/from a column-payload mapping.

    The payload mapping holds the user columns cbrkit owns for a row,
    the key column is added/stripped separately by the backend.  Three
    value shapes are supported, selected at construction:

    * **plain text**, `model` is `None` and `value_column` is set:
      `V` is `str`, the value is stored as `{value_column: value}`
      and read back as the bare string.
    * **mapping**, `model` is `None` and `value_column` is `None`:
      `V` is `Mapping[str, Any]`, the payload is the mapping itself.
    * **typed model**, `model` is a class: Pydantic
      `pydantic.BaseModel` values round-trip via
      `model_dump` / `model_validate`, any other class (dataclass,
      SQLAlchemy mapped class, ...) is read via `getattr` and rebuilt via
      its constructor, with the payload being `columns` projected from the
      value.

    `columns` is the explicit set of cbrkit-owned user columns, driven
    off the resolved schema, never off dataclass `init=` flags, so the
    dump/load round-trip stays symmetric and independent of how the host
    declared the model.

    Args:
        model: Class of typed values, plain text or mappings if None.
        columns: The user columns of a typed value.
        value_column: Column of plain text values, mappings if None and there is no model.
        key_column: Column of the key, which `decode` restores into the field of that name if the model has one,
            so a field that the model would otherwise fill itself, such as a generated id, matches the key.

    Examples:
        >>> import dataclasses
        >>> @dataclasses.dataclass
        ... class Car:
        ...     desc: str
        ...     brand: str
        >>> codec = RowCodec(model=Car, columns=("desc", "brand"))
        >>> codec.encode(Car("red sedan", "audi"))
        {'desc': 'red sedan', 'brand': 'audi'}
        >>> codec.decode({"desc": "red sedan", "brand": "audi"})
        Car(desc='red sedan', brand='audi')
        >>> RowCodec(value_column="value").encode("hello")
        {'value': 'hello'}
    """

    model: type[V] | None = None
    columns: tuple[str, ...] = ()
    value_column: str | None = None
    key_column: str | None = None
    key_field: str | None = field(init=False, default=None)
    init_fields: frozenset[str] | None = field(init=False, default=None)
    """The fields the constructor of a dataclass model takes, None for other models."""

    def __post_init__(self) -> None:
        if self.model is None:
            return

        model = cast(_DataclassOrModel, self.model)

        if self.key_column in model_columns(model):
            object.__setattr__(self, "key_field", self.key_column)

        if is_dataclass(model):
            object.__setattr__(
                self, "init_fields", frozenset(f.name for f in fields(model) if f.init)
            )

    def encode(self, value: V) -> dict[str, Any]:
        """Project a value to its column payload (key column excluded).

        Pydantic models go through `model_dump`, every other model type
        (dataclass, SQLAlchemy mapped class, ...) is read attribute-wise via
        `getattr`, the latter covers ORM rows used as plain data carriers.
        """
        if self.model is None:
            if self.value_column is not None:
                return {self.value_column: value}
            return dict(cast(Mapping[str, Any], value))
        if issubclass(self.model, BaseModel):
            return cast(BaseModel, value).model_dump(include=set(self.columns))
        return {c: getattr(value, c) for c in self.columns}

    def decode(self, payload: Mapping[str, Any], key: Any = None) -> V:
        """Rebuild a value from a column payload (extra columns ignored), with its `key` in the `key_field`.

        Fields of a dataclass that its constructor does not take are set on the built value.
        """
        if self.model is None:
            if self.value_column is not None:
                return cast(V, payload[self.value_column])
            return cast(V, dict(payload))
        data = {c: payload[c] for c in self.columns if c in payload}
        if self.key_field is not None and key is not None:
            data[self.key_field] = key
        if issubclass(self.model, BaseModel):
            return cast(V, cast(type[BaseModel], self.model).model_validate(data))
        init = self.init_fields
        if init is None:
            return self.model(**data)
        value = self.model(**{k: v for k, v in data.items() if k in init})
        for name, item in data.items():
            if name not in init:
                # bypasses a frozen dataclass, while descriptors such as those of an ORM still apply
                object.__setattr__(value, name, item)
        return value


type VectorMetric = Cosine | Dot | Euclidean
"""Vector metric of a dense index, one of `cbrkit.sim.embed.metrics.VECTOR_METRICS`.

The scores come from the backend, so they may differ from those of the metric itself.
"""


__all__ = [
    "RowCodec",
    "VectorMetric",
    "make_codec",
    "model_columns",
]
