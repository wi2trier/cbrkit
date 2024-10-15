from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from cbrkit.helpers import SimPairWrapper, get_metadata
from cbrkit.typing import AnySimFunc, JsonDict, SimPairFunc, SupportsMetadata

__all__ = ["table", "equality", "static"]


def default_key_getter(x: Any) -> Any:
    return x


# TODO: convert to SimSeqFunc
@dataclass(slots=True)
class table[K, V](SimPairFunc[V, float], SupportsMetadata):
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table

    Examples:
        >>> sim = table([("a", "b", 0.5), ("b", "c", 0.7)], symmetric=True, default=0.0)
        >>> sim("b", "a")
        0.5
        >>> sim("a", "c")
        0.0
    """

    symmetric: bool
    default: float | SimPairFunc[V, float]
    key_getter: Callable[[V], K]
    table: dict[tuple[K, K], float | SimPairFunc[V, float]]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": get_metadata(self.default)
            if isinstance(self.default, Callable)
            else self.default,
            "key_getter": get_metadata(self.key_getter),
            "table": [
                {
                    "x": str(k[0]),
                    "y": str(k[1]),
                    "value": get_metadata(v) if isinstance(v, Callable) else v,
                }
                for k, v in self.table.items()
            ],
        }

    def __init__(
        self,
        entries: Sequence[tuple[K, K, float | AnySimFunc[V, float]]]
        | Mapping[tuple[K, K], float | AnySimFunc[V, float]],
        symmetric: bool = True,
        default: float | AnySimFunc[V, float] = 0.0,
        key_getter: Callable[[V], K] | None = None,
    ):
        self.default = (
            SimPairWrapper(default) if isinstance(default, Callable) else default
        )
        self.symmetric = symmetric
        self.table = {}
        self.key_getter = key_getter or default_key_getter

        if isinstance(entries, Mapping):
            for (x, y), raw_value in entries.items():
                value = (
                    SimPairWrapper(raw_value)
                    if isinstance(raw_value, Callable)
                    else raw_value
                )
                self.table[(x, y)] = value

                if self.symmetric:
                    self.table[(y, x)] = value
        else:
            for entry in entries:
                x, y, raw_value = entry
                value = (
                    SimPairWrapper(raw_value)
                    if isinstance(raw_value, Callable)
                    else raw_value
                )
                self.table[(x, y)] = value

                if self.symmetric:
                    self.table[(y, x)] = value

    @override
    def __call__(self, x: V, y: V) -> float:
        x_key = self.key_getter(x)
        y_key = self.key_getter(y)

        entry = self.table.get((x_key, y_key), self.default)

        if isinstance(entry, Callable):
            return entry(x, y)

        return entry


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
