from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from cbrkit.helpers import SimSeqWrapper, get_metadata
from cbrkit.typing import (
    AnySimFunc,
    Float,
    JsonDict,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
    SupportsMetadata,
)

__all__ = ["table", "static_table", "dynamic_table", "equality", "static"]


@dataclass(slots=True)
class static_table[V](SimPairFunc[V, float], SupportsMetadata):
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

    default: float
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
        default: float = 0.0,
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
    def __call__(self, x: V, y: V) -> float:
        return self.table.get((x, y), self.default)


table = static_table


def default_key_getter(x: Any) -> Any:
    return x


@dataclass(slots=True)
class dynamic_table[K, V, S: Float](SimSeqFunc[V, S], SupportsMetadata):
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table

    Examples:
        >>> sim = dynamic_table(
        ...     [("a", "b", static(0.5)), ("b", "c", static(0.7))],
        ...     symmetric=True,
        ...     default=static(0.0)
        ... )
        >>> sim([("b", "a"), ("a", "c")])
        [0.5, 0.0]
    """

    symmetric: bool
    default: SimSeqFunc[V, S]
    key_getter: Callable[[V], K]
    table: dict[tuple[K, K], SimSeqFunc[V, S]]

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
        entries: Sequence[tuple[K, K, AnySimFunc[V, S]]]
        | Mapping[tuple[K, K], AnySimFunc[V, S]],
        default: AnySimFunc[V, S],
        symmetric: bool = True,
        key_getter: Callable[[V], K] = default_key_getter,
    ):
        self.default = SimSeqWrapper(default)
        self.symmetric = symmetric
        self.key_getter = key_getter
        self.table = {}

        if isinstance(entries, Mapping):
            for (x, y), val in entries.items():
                func = SimSeqWrapper(val)
                self.table[(x, y)] = func

                if self.symmetric:
                    self.table[(y, x)] = func
        else:
            for entry in entries:
                x, y, val = entry
                func = SimSeqWrapper(val)
                self.table[(x, y)] = func

                if self.symmetric:
                    self.table[(y, x)] = func

    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> SimSeq[S]:
        # first we get the keys for each pair
        keys = [(self.key_getter(x), self.key_getter(y)) for x, y in pairs]

        # then we group the pairs by key to avoid redundant computations
        idx_map: defaultdict[tuple[K, K] | None, list[int]] = defaultdict(list)

        for idx, key in enumerate(keys):
            if key in self.table:
                idx_map[key].append(idx)
            else:
                idx_map[None].append(idx)

        # now we compute the similarities
        results: dict[int, S] = {}

        for key, idxs in idx_map.items():
            sim_func = self.default

            if key is not None:
                sim_func = self.table[key]

            sims = sim_func([pairs[idx] for idx in idxs])

            for idx, sim in zip(idxs, sims, strict=True):
                results[idx] = sim

        return [results[idx] for idx in range(len(pairs))]


@dataclass(slots=True, frozen=True)
class equality(SimPairFunc[Any, float], SupportsMetadata):
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
class static(SimPairFunc[Any, float], SupportsMetadata):
    """Static similarity function. Returns a constant value for all pairs.

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
