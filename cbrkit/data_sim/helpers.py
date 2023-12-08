from cbrkit.typing import (
    DataPairSimFunc,
    DataSimFunc,
    DataType,
    SimilaritySequence,
)


def apply(func: DataPairSimFunc[DataType]) -> DataSimFunc[DataType]:
    def wrapped_func(*args: tuple[DataType, DataType]) -> SimilaritySequence:
        return [func(data1, data2) for (data1, data2) in args]

    return wrapped_func


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)
