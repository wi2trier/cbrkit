from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from cbrkit.typing import (
    SimPairFunc,
    SimVal,
    ValueType,
)


def table(
    entries: Sequence[tuple[ValueType, ValueType, SimVal]],
    symmetric: bool = True,
    default: SimVal = 0.0,
) -> SimPairFunc[ValueType]:
    table: defaultdict[ValueType, defaultdict[ValueType, SimVal]] = defaultdict(
        lambda: defaultdict(lambda: default)
    )

    for x in entries:
        table[x[0]][x[1]] = x[2]

        if symmetric:
            table[x[1]][x[0]] = x[2]

    def wrapped_func(x: ValueType, y: ValueType) -> SimVal:
        return table[x][y]

    return wrapped_func


def equality() -> SimPairFunc[Any]:
    def wrapped_func(x: Any, y: Any) -> SimVal:
        return x == y

    return wrapped_func
