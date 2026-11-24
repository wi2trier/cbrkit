"""Internal utilities shared by the modules of cbrkit, not part of the public API."""

from collections.abc import Awaitable, Callable, Collection, Iterable, Sequence
from heapq import nlargest
from operator import itemgetter
from typing import Any, cast
from uuid import uuid7

from .errors import Conflict, Refused
from .typing import AttributeGetter, Casebase, Float, HasValue, SimMap

RRF_K = 60
"""Smoothing constant of reciprocal rank fusion, larger values narrow the gap between top ranks."""

RETRIES = 2
"""How often a run is planned again when the storage changed before its changes were applied."""


def get_name(obj: object) -> str:
    """The name of an object, its `__name__` or else the name of its class, a string as it is, and empty for None.

    Examples:
        >>> import functools
        >>> get_name(len), get_name(functools.partial(len)), get_name(None)
        ('len', 'partial', '')
    """
    if obj is None:
        return ""

    if isinstance(obj, str):
        return obj

    name = getattr(obj, "__name__", None)

    return name if isinstance(name, str) else type(obj).__name__


def compute_index_diff[K, V](
    existing: Casebase[K, V],
    data: Casebase[K, V],
) -> tuple[set[K], Casebase[K, V]]:
    """Compute stale keys and changed/new entries for index updates.

    Args:
        existing: The current indexed casebase.
        data: The desired casebase to sync with.

    Returns:
        A tuple `(stale_keys, changed_or_new)` where `stale_keys` are
        keys present in `existing` but absent from `data`, and
        `changed_or_new` maps keys whose values differ or are new.
    """
    new_keys = set(data.keys())
    old_keys = set(existing.keys())
    stale_keys = old_keys - new_keys
    changed_or_new: Casebase[K, V] = {
        k: data[k] for k in new_keys if k not in existing or existing[k] != data[k]
    }
    return stale_keys, changed_or_new


def sorted_keys[K](keys: Iterable[K]) -> list[K]:
    """Keys in a deterministic order, which for integer and UUIDv7 keys is the order of creation.

    Keys are grouped by the name of their type and ordered within it,
    by their representation if they are not orderable.
    Set iteration order depends on the hash seed of the process,
    so it must never decide between equally good choices.

    Examples:
        >>> sorted_keys([2, "b", 1, "a"])
        [1, 2, 'a', 'b']
        >>> sorted_keys([(1, "a"), (1, 2)])
        [(1, 'a'), (1, 2)]
    """
    items = list(keys)

    try:
        return sorted(items, key=lambda key: (type(key).__name__, key))
    except TypeError:
        return sorted(items, key=lambda key: (type(key).__name__, repr(key)))


def next_keys[K](taken: Collection[K], count: int, /) -> list[K]:
    """The keys of new cases: the integers after the largest key if every key is an integer, new UUIDv7 strings otherwise.

    An empty storage starts at the integer 0, so a storage of other keys passes its own `new_keys`.

    Examples:
        >>> next_keys([0, 4, 2], 2), next_keys([], 1)
        ([5, 6], [0])
        >>> [len(key) for key in next_keys(["a"], 1)]
        [36]
    """
    ints = [key for key in taken if isinstance(key, int)]

    if len(ints) == len(taken):
        start = max(ints, default=-1) + 1

        return cast(list[K], list(range(start, start + count)))

    return cast(list[K], [str(uuid7()) for _ in range(count)])


def check_expected[K, V](
    current: Casebase[K, V], expected: Casebase[K, V] | None
) -> None:
    """Raises `Conflict` unless a storage holds the cases a mutation was planned on.

    Examples:
        >>> check_expected({0: "a"}, {0: "a"})
        >>> check_expected({0: "a"}, {0: "b"})
        Traceback (most recent call last):
        ...
        cbrkit.errors.Conflict: The cases changed since the mutation was planned
    """
    if (
        expected is not None
        and current is not expected
        and dict(current) != dict(expected)
    ):
        raise Conflict("The cases changed since the mutation was planned")


async def retry_conflicts[T](attempt: Callable[[], Awaitable[T]]) -> T:
    """Runs an attempt again up to `RETRIES` times while it raises `Conflict`, such as a run planned on changed cases.

    Examples:
        >>> import asyncio
        >>> attempts = iter([Conflict("changed"), Conflict("changed"), "applied"])
        >>> async def attempt() -> str:
        ...     result = next(attempts)
        ...     if isinstance(result, Conflict):
        ...         raise result
        ...     return result
        >>> asyncio.run(retry_conflicts(attempt))
        'applied'
    """
    for _ in range(RETRIES):
        try:
            return await attempt()
        except Conflict:
            pass

    return await attempt()


def check_absent[K](stored: Collection[K], keys: Iterable[K]) -> None:
    """Raises `Conflict` if a storage holds a case under a key that a mutation inserts.

    Examples:
        >>> check_absent({0: "a"}, [1])
        >>> check_absent({0: "a"}, [0])
        Traceback (most recent call last):
        ...
        cbrkit.errors.Conflict: 0 exists already
    """
    if existing := [key for key in keys if key in stored]:
        raise Conflict(f"{existing[0]!r} exists already")


def merge_patch[K, V](
    upsert: Casebase[K, V] | None,
    insert: Casebase[K, V] | None,
    delete: Collection[K] | None,
) -> tuple[dict[K, V], set[K]]:
    """The cases that a patch writes under their keys and the keys that it deletes.

    Raises:
        ValueError: If a key is both upserted and inserted, or written and deleted.

    Examples:
        >>> merge_patch({0: "a"}, {1: "b"}, [2])
        ({0: 'a', 1: 'b'}, {2})
        >>> merge_patch({0: "a"}, None, [0])
        Traceback (most recent call last):
        ...
        ValueError: A patch must change a case once, but changes {0}
    """
    upserted = dict(upsert or {})
    inserted = dict(insert or {})
    deleted = set(delete or ())

    if overlap := (upserted.keys() & inserted.keys()) | (
        (upserted.keys() | inserted.keys()) & deleted
    ):
        raise ValueError(f"A patch must change a case once, but changes {overlap!r}")

    return upserted | inserted, deleted


def plan_patch[K, V](
    stored: Casebase[K, V],
    upsert: Casebase[K, V] | None,
    delete: Collection[K] | None,
    add: Sequence[V],
    insert: Casebase[K, V] | None,
    expected: Casebase[K, V] | None,
    new_keys: Callable[[Collection[K], int], list[K]] = next_keys,
) -> tuple[dict[K, V], set[K], list[K]]:
    """The cases that a patch of the stored cases writes, the added ones under new keys, the keys that it deletes, and the new keys.

    Raises:
        ValueError: If a key is both upserted and inserted, or written and deleted.
        cbrkit.errors.Conflict: If the stored cases are not the expected ones or hold a key to insert.

    Examples:
        >>> plan_patch({0: "a"}, {0: "b"}, None, ["c"], None, {0: "a"})
        ({0: 'b', 1: 'c'}, set(), [1])
    """
    written, deleted = merge_patch(upsert, insert, delete)
    check_expected(stored, expected)
    check_absent(stored, insert or ())
    new = new_keys(stored.keys() | written.keys(), len(add)) if add else []
    written.update(zip(new, add, strict=True))

    return written, deleted, new


def patched[K, V](
    cases: Casebase[K, V], written: Casebase[K, V], deleted: Collection[K]
) -> dict[K, V]:
    """The cases after a patch that writes and deletes cases, see `plan_patch`."""
    result = {key: case for key, case in cases.items() if key not in deleted}
    result.update(written)

    return result


def get_value[T](obj: HasValue[T]) -> T:
    """Return the value held by an object such as a graph node."""
    return obj.value


def sim_map2ranking[K, S: Float](
    similarities: SimMap[K, S],
    limit: int | None = None,
    min_similarity: float | None = None,
) -> list[K]:
    """Sort similarity map keys by descending similarity score, ties in map order.

    Args:
        similarities: Similarity by key.
        limit: Maximal number of keys, all if None.
        min_similarity: Minimal similarity of a kept key, any if None.

    Examples:
        >>> sim_map2ranking({"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.1}, 3, 0.2)
        ['b', 'a', 'c']
    """
    scored = [
        (key, float(similarity))
        for key, similarity in similarities.items()
        if min_similarity is None or float(similarity) >= min_similarity
    ]
    ranked = (
        sorted(scored, key=itemgetter(1), reverse=True)
        if limit is None
        else nlargest(limit, scored, key=itemgetter(1))
    )

    return [key for key, _ in ranked]


def identity[T](x: T) -> T:
    """Return the argument unchanged."""
    return x


def fold[T](value: T, case_sensitive: bool) -> T | str:
    """A value as compared, case-folded if it is a string and the comparison is not case-sensitive."""
    return value.casefold() if isinstance(value, str) and not case_sensitive else value


def getitem_or_getattr(obj: Any, key: Any) -> Any:
    """Retrieve a value by key using item access or attribute access as fallback."""
    if hasattr(obj, "__getitem__"):
        return obj[key]

    return getattr(obj, key)


def get_attribute(
    obj: Any, name: str, getter: AttributeGetter = getitem_or_getattr
) -> Any | None:
    """The value of an attribute, None if it is missing.

    Args:
        obj: The value whose attribute is read, such as a case.
        name: The name of the attribute.
        getter: Reads the attribute, a `LookupError`, `AttributeError`, or `TypeError` means it is missing.

    Examples:
        >>> get_attribute({"price": 10}, "price"), get_attribute({"price": 10}, "year"), get_attribute(None, "price")
        (10, None, None)
    """
    try:
        return getter(obj, name)
    except LookupError, AttributeError, TypeError:
        return None


def keyed[T](rows: Iterable[T], key_column: str | None, start: int = 0) -> dict[Any, T]:
    """Records keyed by the value of their `key_column` field, or by position if it is None.

    Examples:
        >>> keyed([{"id": "a"}, {"id": "b"}], "id")
        {'a': {'id': 'a'}, 'b': {'id': 'b'}}
    """
    if key_column is None:
        return dict(enumerate(rows, start))

    data: dict[Any, T] = {}

    for row in rows:
        try:
            value = getitem_or_getattr(row, key_column)

            if value in data:
                raise Refused(f"Duplicate key: {value!r}")
        except (LookupError, AttributeError, TypeError) as exc:
            raise Refused(f"A record has no usable key {key_column!r}") from exc

        data[value] = row

    return data


def fuse_rankings[K, V, S: Float](
    results: Iterable[tuple[Casebase[K, V], SimMap[K, S]]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
    limit: int | None = None,
) -> tuple[dict[K, V], dict[K, float]]:
    """Fuse the rankings of several results by weighted reciprocal rank fusion, see `cbrkit.retrieve.fuse`."""
    scores: dict[K, float] = {}
    values: dict[K, V] = {}
    weighted = (
        zip(results, weights, strict=True)
        if weights is not None
        else ((result, 1.0) for result in results)
    )

    for (casebase, similarities), weight in weighted:
        for rank, key in enumerate(sim_map2ranking(similarities), start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            values.setdefault(key, casebase[key])

    ranking = sim_map2ranking(scores, limit)

    return {key: values[key] for key in ranking}, {key: scores[key] for key in ranking}
