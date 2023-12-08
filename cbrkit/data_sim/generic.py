from collections import defaultdict
from typing import Any

from cbrkit.data_sim.helpers import apply
from cbrkit.typing import (
    DataSimFunc,
    DataType,
    SimilaritySequence,
    SimilarityValue,
)


def table(
    *args: tuple[DataType, DataType, SimilarityValue],
    symmetric: bool = True,
    default: SimilarityValue = 0.0,
) -> DataSimFunc[DataType]:
    table: defaultdict[DataType, defaultdict[DataType, SimilarityValue]] = defaultdict(
        lambda: defaultdict(lambda: default)
    )

    for arg in args:
        table[arg[0]][arg[1]] = arg[2]

        if symmetric:
            table[arg[1]][arg[0]] = arg[2]

    def wrapped_func(*args: tuple[DataType, DataType]) -> SimilaritySequence:
        return [table[arg[0]][arg[1]] for arg in args]

    return wrapped_func


def equality() -> DataSimFunc[Any]:
    @apply
    def wrapped_func(x: Any, y: Any) -> SimilarityValue:
        return x == y

    return wrapped_func
