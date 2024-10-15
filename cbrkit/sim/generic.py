from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, override

from cbrkit.typing import JsonDict, SimPairFunc, SupportsMetadata

__all__ = ["table", "equality", "static"]


@dataclass(slots=True)
class table[V](SimPairFunc[V, float], SupportsMetadata):
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

    entries: Sequence[tuple[V, V, float]]
    symmetric: bool = True
    default: float = 0.0
    table: defaultdict[V, defaultdict[V, float]] = field(init=False)

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "symmetric": self.symmetric,
            "default": self.default,
        }

    def __post_init__(self):
        self.table = defaultdict(lambda: defaultdict(lambda: self.default))

        for x in self.entries:
            self.table[x[0]][x[1]] = x[2]

            if self.symmetric:
                self.table[x[1]][x[0]] = x[2]

    @override
    def __call__(self, x: V, y: V) -> float:
        return self.table[x][y]


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
