from collections import defaultdict

from cbrkit import model


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
