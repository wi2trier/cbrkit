from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from cbrkit.typing import (
    SimPairFunc,
    ValueType,
)


def table(
    entries: Sequence[tuple[ValueType, ValueType, float]],
    symmetric: bool = True,
    default: float = 0.0,
) -> SimPairFunc[ValueType, float]:
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
    def wrapped_func(x: Any, y: Any) -> float:
        return 1.0 if x == y else 0.0

    return wrapped_func
