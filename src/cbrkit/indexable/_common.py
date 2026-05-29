"""Shared helpers and SQL utilities for indexable storage backends."""

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel

from ..helpers import dist2sim
from ..typing import Casebase, DataclassOrModel


def model_columns(model: DataclassOrModel) -> tuple[str, ...]:
    """Return the declared field names of a dataclass or pydantic model."""
    if is_dataclass(model):
        return tuple(f.name for f in fields(model))
    return tuple(cast(type[BaseModel], model).model_fields)


def make_codec[V](
    model: type[V] | None,
    value_column: str,
    *,
    key_column: str | None = None,
) -> "RowCodec[V]":
    """Build a :class:`RowCodec` for a storage backend.

    The casebase key is never part of the codec payload — this is the
    single, uniform policy across backends.  How the key is kept out
    depends on where the backend stores it:

    * **column-oriented** backends (e.g. ``lancedb``) keep the key in a
      dedicated *key_column* that shares the column namespace with the
      model's fields, so it is excluded from the payload here.
    * **record-oriented** backends (``chromadb``, ``zvec``) store the key
      as the native record id, a separate namespace, and pass
      ``key_column=None`` since no model column can collide with it.

    Without a model the value is plain text stored under *value_column*.
    The resulting ``codec.columns`` is the authoritative list of
    cbrkit-owned columns, so backends read it directly instead of
    recomputing it.
    """
    if model is None:
        columns = (value_column,)
    else:
        names = model_columns(cast(DataclassOrModel, model))
        columns = tuple(c for c in names if c != key_column)
    return RowCodec(model=model, columns=columns, value_column=value_column)


@dataclass(slots=True, frozen=True)
class RowCodec[V]:
    """Convert a casebase value ``V`` to/from a column-payload mapping.

    The payload mapping holds the *user* columns cbrkit owns for a row;
    the key column is added/stripped separately by the backend.  Three
    value shapes are supported, selected at construction:

    * **plain text** — ``model`` is ``None`` and ``value_column`` is set:
      ``V`` is ``str``; the value is stored as ``{value_column: value}``
      and read back as the bare string.
    * **mapping** — ``model`` is ``None`` and ``value_column`` is ``None``:
      ``V`` is ``Mapping[str, Any]``; the payload is the mapping itself.
    * **typed model** — ``model`` is a class: Pydantic
      :class:`~pydantic.BaseModel` values round-trip via
      ``model_dump`` / ``model_validate``; any other class (dataclass,
      SQLAlchemy mapped class, ...) is read via ``getattr`` and rebuilt via
      its constructor, with the payload being ``columns`` projected from the
      value.

    ``columns`` is the explicit set of cbrkit-owned user columns — driven
    off the resolved schema, never off dataclass ``init=`` flags — so the
    dump/load round-trip stays symmetric and independent of how the host
    declared the model.

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

    def encode(self, value: V) -> dict[str, Any]:
        """Project a value to its column payload (key column excluded).

        Pydantic models go through ``model_dump``; every other model type
        (dataclass, SQLAlchemy mapped class, ...) is read attribute-wise via
        ``getattr`` — the latter covers ORM rows used as plain data carriers.
        """
        if self.model is None:
            if self.value_column is not None:
                return {self.value_column: value}
            return dict(cast(Mapping[str, Any], value))
        if issubclass(self.model, BaseModel):
            return cast(BaseModel, value).model_dump(include=set(self.columns))
        return {c: getattr(value, c) for c in self.columns}

    def decode(self, payload: Mapping[str, Any]) -> V:
        """Rebuild a value from a column payload (extra columns ignored)."""
        if self.model is None:
            if self.value_column is not None:
                return cast(V, payload[self.value_column])
            return cast(V, dict(payload))
        data = {c: payload[c] for c in self.columns if c in payload}
        if issubclass(self.model, BaseModel):
            return cast(V, cast(type[BaseModel], self.model).model_validate(data))
        return self.model(**data)


def _compute_index_diff[K, V](
    existing: Casebase[K, V],
    data: Casebase[K, V],
) -> tuple[set[K], Casebase[K, V]]:
    """Compute stale keys and changed/new entries for index updates.

    Args:
        existing: The current indexed casebase.
        data: The desired casebase to sync with.

    Returns:
        A tuple `(stale_keys, changed_or_new)` where *stale_keys* are
        keys present in *existing* but absent from *data*, and
        *changed_or_new* maps keys whose values differ or are new.
    """
    new_keys = set(data.keys())
    old_keys = set(existing.keys())
    stale_keys = old_keys - new_keys
    changed_or_new: Casebase[K, V] = {
        k: data[k] for k in new_keys if k not in existing or existing[k] != data[k]
    }
    return stale_keys, changed_or_new


def _normalize_patch_keys[K](
    upsert: Collection[K] | None,
    delete: Collection[K] | None,
) -> tuple[set[K], set[K]] | None:
    """Validate `patch_index` arguments and split into key sets.

    Mappings count as collections of their keys, so this helper works
    for both keyed (`Casebase[K, str]`) and bare (`Collection[str]`)
    patch arguments.

    Returns `None` when both inputs are empty (the patch is a no-op).
    Raises `ValueError` if the same key appears in both *upsert* and
    *delete*.
    """
    if not upsert and not delete:
        return None

    upsert_keys = set(upsert) if upsert else set()
    delete_keys = set(delete) if delete else set()
    overlap = upsert_keys & delete_keys

    if overlap:
        raise ValueError(f"Cannot upsert and delete the same entries: {overlap!r}")

    return upsert_keys, delete_keys


def _sql_literal(value: int | str) -> str:
    """Return a SQL literal for supported index keys."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"

    return str(value)


def _sql_in_clause[K: int | str](column: str, keys: Collection[K]) -> str:
    """Build a SQL `IN (...)` predicate for supported index keys."""
    return f"{column} IN ({', '.join(_sql_literal(k) for k in keys)})"


@dataclass(slots=True, frozen=True)
class _PgMetric:
    """Per-metric pgvector configuration."""

    opclass: str
    """HNSW operator class name passed to `CREATE INDEX ... USING hnsw`."""
    distance_attr: str
    """`pgvector.sqlalchemy.Vector` method name returning the distance expression."""
    sim_from_distance: Callable[[float], float]
    """Convert a raw pgvector distance to a similarity score."""


PG_METRICS: dict[Literal["cosine", "ip", "l2"], _PgMetric] = {
    "cosine": _PgMetric("vector_cosine_ops", "cosine_distance", dist2sim),
    "ip": _PgMetric("vector_ip_ops", "max_inner_product", lambda d: -d),
    "l2": _PgMetric("vector_l2_ops", "l2_distance", dist2sim),
}


__all__ = [
    "_compute_index_diff",
    "_normalize_patch_keys",
    "_sql_literal",
    "_sql_in_clause",
    "_PgMetric",
    "PG_METRICS",
    "RowCodec",
    "make_codec",
    "model_columns",
]
