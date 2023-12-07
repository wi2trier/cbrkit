from collections import defaultdict
from typing import Any

from cbrkit import model
from cbrkit.data_sim.helpers import apply


def table(
    *args: tuple[model.DataType, model.DataType, model.SimilarityValue],
    symmetric: bool = True,
    default: model.SimilarityValue = 0.0,
) -> model.DataSimilarityBatchFunc[model.DataType]:
    table: defaultdict[
        model.DataType, defaultdict[model.DataType, model.SimilarityValue]
    ] = defaultdict(lambda: defaultdict(lambda: default))

    for arg in args:
        table[arg[0]][arg[1]] = arg[2]

        if symmetric:
            table[arg[1]][arg[0]] = arg[2]

    def wrapped_func(
        *args: tuple[model.DataType, model.DataType]
    ) -> model.SimilaritySequence:
        return [table[arg[0]][arg[1]] for arg in args]

    return wrapped_func


def equality() -> model.DataSimilarityBatchFunc[Any]:
    @apply
    def wrapped_func(x: Any, y: Any) -> model.SimilarityValue:
        return x == y

    return wrapped_func
