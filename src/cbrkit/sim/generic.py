from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, override

from ..helpers import batchify_sim, get_metadata
from ..typing import (
    AnySimFunc,
    BatchSimFunc,
    Float,
    HasMetadata,
    JsonDict,
    SimFunc,
    SimSeq,
)

__all__ = [
    "table",
    "static_table",
    "dynamic_table",
    "type_table",
    "equality",
    "type_equality",
    "static",
]


@dataclass(slots=True)
class static_table[V](SimFunc[V | Any, float], HasMetadata):
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        default: Default similarity value for pairs not in the table
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)

    Examples:
        >>> sim = static_table(
        ...     [("a", "b", 0.5), ("b", "c", 0.7)],
        ...     symmetric=True,
        ...     default=0.0
        ... )
        >>> sim("b", "a")
        0.5
        >>> sim("a", "c")
        0.0
    """

    default: float | None
    symmetric: bool
    table: dict[tuple[V, V], float]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": self.default,
            "table": [
                {
                    "x": str(k[0]),
                    "y": str(k[1]),
                    "value": v,
                }
                for k, v in self.table.items()
            ],
        }

    def __init__(
        self,
        entries: Sequence[tuple[V, V, float]] | Mapping[tuple[V, V], float],
        default: float | None = None,
        symmetric: bool = True,
    ):
        self.default = default
        self.symmetric = symmetric
        self.table = {}

        if isinstance(entries, Mapping):
            for (x, y), val in entries.items():
                self.table[(x, y)] = val

                if self.symmetric:
                    self.table[(y, x)] = val
        else:
            for entry in entries:
                x, y, val = entry
                self.table[(x, y)] = val

                if self.symmetric:
                    self.table[(y, x)] = val

    @override
    def __call__(self, x: V | Any, y: V | Any) -> float:
        sim = self.table.get((x, y), self.default)

        if sim is None:
            raise ValueError(f"Pair ({x}, {y}) not in the table")

        return sim


table = static_table


@dataclass(slots=True)
class dynamic_table[K, U, V, S: Float](BatchSimFunc[U | V, S], HasMetadata):
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table
        key_getter: A function that extracts the the key for lookup from the input values

    Examples:
        >>> from cbrkit.helpers import identity
        >>> sim = dynamic_table(
        ...     {
        ...         ("a", "b"): static(0.5),
        ...         ("b", "c"): static(0.7)
        ...     },
        ...     symmetric=True,
        ...     default=static(0.0),
        ...     key_getter=identity,
        ... )
        >>> sim([("b", "a"), ("a", "c")])
        [0.5, 0.0]
    """

    symmetric: bool
    default: BatchSimFunc[U, S] | None
    key_getter: Callable[[U | V], K]
    table: dict[tuple[K, K], BatchSimFunc[V, S]]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": get_metadata(self.default),
            "key_getter": get_metadata(self.key_getter),
            "table": [
                {
                    "x": str(k[0]),
                    "y": str(k[1]),
                    "value": get_metadata(v),
                }
                for k, v in self.table.items()
            ],
        }

    def __init__(
        self,
        entries: Mapping[tuple[K, K], AnySimFunc[..., S]]
        | Mapping[K, AnySimFunc[..., S]],
        key_getter: Callable[[U | V], K],
        default: AnySimFunc[U, S] | None = None,
        symmetric: bool = True,
    ):
        self.default = batchify_sim(default) if default is not None else None
        self.symmetric = symmetric
        self.key_getter = key_getter
        self.table = {}

        for key, val in entries.items():
            func = batchify_sim(val)

            if isinstance(key, tuple):
                x, y = cast(tuple[K, K], key)
            else:
                x = y = cast(K, key)

            self.table[(x, y)] = func

            if self.symmetric and x != y:
                self.table[(y, x)] = func

    @override
    def __call__(self, batches: Sequence[tuple[U | V, U | V]]) -> SimSeq[S]:
        # then we group the batches by key to avoid redundant computations
        idx_map: defaultdict[tuple[K, K] | None, list[int]] = defaultdict(list)

        for idx, (x, y) in enumerate(batches):
            key = (self.key_getter(x), self.key_getter(y))

            if key in self.table:
                idx_map[key].append(idx)
            else:
                idx_map[None].append(idx)

        # now we compute the similarities
        results: dict[int, S] = {}

        for key, idxs in idx_map.items():
            sim_func = cast(
                BatchSimFunc[U | V, S],
                self.table.get(key) if key is not None else self.default,
            )

            if sim_func is None:
                missing_entries = [batches[idx] for idx in idxs]
                missing_keys = {
                    (self.key_getter(x), self.key_getter(y)) for x, y in missing_entries
                }

                raise ValueError(f"Pairs {missing_keys} not in the table")

            sims = sim_func([batches[idx] for idx in idxs])

            for idx, sim in zip(idxs, sims, strict=True):
                results[idx] = sim

        return [results[idx] for idx in range(len(batches))]


def type_table[U, V, S: Float](
    entries: Mapping[type[V], AnySimFunc[..., S]],
    default: AnySimFunc[U, S] | None = None,
) -> BatchSimFunc[U | V, S]:
    return dynamic_table(
        entries=entries,
        key_getter=type,
        default=default,
        symmetric=False,
    )


@dataclass(slots=True, frozen=True)
class equality(SimFunc[Any, float]):
    """Equality similarity function. Returns 1.0 if the two values are equal, 0.0 otherwise.

    Examples:
        >>> sim = equality()
        >>> sim("b", "a")
        0.0
        >>> sim("a", "a")
        1.0
    """

    @override
    def __call__(self, x: Any, y: Any) -> float:
        return 1.0 if x == y else 0.0


@dataclass(slots=True, frozen=True)
class type_equality(SimFunc[Any, float]):
    """Type equality similarity function. Returns 1.0 if the two values have the same type, 0.0 otherwise.

    Examples:
        >>> sim = type_equality()
        >>> sim("b", "a")
        1.0
        >>> sim("a", 1)
        0.0
    """

    @override
    def __call__(self, x: Any, y: Any) -> float:
        return 1.0 if type(x) is type(y) else 0.0


@dataclass(slots=True, frozen=True)
class static(SimFunc[Any, float]):
    """Static similarity function. Returns a constant value for all batches.

    Args:
        value: The constant similarity value

    Examples:
        >>> sim = static(0.5)
        >>> sim("b", "a")
        0.5
        >>> sim("a", "a")
        0.5
    """

    value: float

    @override
    def __call__(self, x: Any, y: Any) -> float:
        return self.value
