from collections import defaultdict
from typing import Any

from cbrkit.typing import (
    SimFunc,
    SimilarityValue,
    ValueType,
)


def table(
    *args: tuple[ValueType, ValueType, SimilarityValue],
    symmetric: bool = True,
    default: SimilarityValue = 0.0,
) -> SimFunc[ValueType]:
    table: defaultdict[
        ValueType, defaultdict[ValueType, SimilarityValue]
    ] = defaultdict(lambda: defaultdict(lambda: default))

    for arg in args:
        table[arg[0]][arg[1]] = arg[2]

        if symmetric:
            table[arg[1]][arg[0]] = arg[2]

    def wrapped_func(x: ValueType, y: ValueType) -> SimilarityValue:
        return table[x][y]

    return wrapped_func


def equality() -> SimFunc[Any]:
    def wrapped_func(x: Any, y: Any) -> SimilarityValue:
        return x == y

    return wrapped_func
