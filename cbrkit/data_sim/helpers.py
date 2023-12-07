from cbrkit import model


def apply(
    func: model.DataSimilaritySingleFunc[model.DataType]
) -> model.DataSimilarityBatchFunc[model.DataType]:
    def wrapped_func(
        *args: tuple[model.DataType, model.DataType]
    ) -> model.SimilaritySequence:
        return [func(data1, data2) for (data1, data2) in args]

    return wrapped_func


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)
