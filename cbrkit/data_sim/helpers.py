from cbrkit import model


# TODO: Document that this shall be applied when using a DataSimilaritySingleFunc
def apply(
    func: model.DataSimilaritySingleFunc[model.DataType]
) -> model.DataSimilarityBatchFunc[model.DataType]:
    def wrapped_func(
        *args: tuple[model.DataType, model.DataType]
    ) -> model.SimilaritySequence:
        return [func(data1, data2) for (data1, data2) in args]

    return wrapped_func
