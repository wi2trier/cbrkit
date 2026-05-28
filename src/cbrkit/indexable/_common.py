"""Shared helpers and SQL utilities for indexable storage backends."""

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Literal

from ..helpers import dist2sim
from ..typing import Casebase


def _compute_index_diff[K](
    existing: Casebase[K, str],
    data: Casebase[K, str],
) -> tuple[set[K], Casebase[K, str]]:
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
    changed_or_new: Casebase[K, str] = {
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
]
