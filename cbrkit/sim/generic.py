from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from cbrkit.typing import (
    SimPairFunc,
    ValueType,
)

__all__ = ["table", "equality"]


def table(
    entries: Sequence[tuple[ValueType, ValueType, float]],
    symmetric: bool = True,
    default: float = 0.0,
) -> SimPairFunc[ValueType, float]:
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

    table: defaultdict[ValueType, defaultdict[ValueType, float]] = defaultdict(
        lambda: defaultdict(lambda: default)
    )

    for x in entries:
        table[x[0]][x[1]] = x[2]

        if symmetric:
            table[x[1]][x[0]] = x[2]

    def wrapped_func(x: ValueType, y: ValueType) -> float:
        return table[x][y]

    return wrapped_func


def equality() -> SimPairFunc[Any, float]:
    """Equality similarity function. Returns 1.0 if the two values are equal, 0.0 otherwise.

    Examples:
        >>> sim = equality()
        >>> sim("b", "a")
        0.0
        >>> sim("a", "a")
        1.0
    """

    def wrapped_func(x: Any, y: Any) -> float:
        return 1.0 if x == y else 0.0

    return wrapped_func
