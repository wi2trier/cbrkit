from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from ..helpers import get_metadata, unbatchify_sim
from ..typing import (
    AnySimFunc,
    Float,
    HasMetadata,
    JsonDict,
    SimFunc,
)

__all__ = [
    "table",
    "static_table",
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

    default: SimFunc[V | Any, float] | None
    symmetric: bool
    table: dict[tuple[V, V], float]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": get_metadata(self.default),
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
        default: AnySimFunc[V | Any, float] | float | None = None,
        symmetric: bool = True,
    ):
        self.symmetric = symmetric
        self.table = {}

        if isinstance(default, Callable):
            self.default = unbatchify_sim(default)
        elif default is None:
            self.default = None
        else:
            self.default = static(default)

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
        sim = self.table.get((x, y))

        if sim is None and self.default is not None:
            sim = self.default(x, y)

        if sim is None:
            raise ValueError(f"Pair ({x}, {y}) not in the table")

        return sim


table = static_table


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
class static[S: Float](SimFunc[Any, S]):
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

    value: S

    @override
    def __call__(self, x: Any, y: Any) -> S:
        return self.value
