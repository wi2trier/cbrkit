import math

from cbrkit import model
from cbrkit.data_sim.helpers import apply

Number = float | int


def linear(max: float, min: float = 0.0) -> model.DataSimilarityBatchFunc[Number]:
    @apply
    def wrapped_func(x: Number, y: Number) -> model.SimilarityValue:
        return (max - abs(x - y)) / (max - min)

    return wrapped_func


def threshold(threshold: float) -> model.DataSimilarityBatchFunc[Number]:
    @apply
    def wrapped_func(x: Number, y: Number) -> model.SimilarityValue:
        return 1.0 if abs(x - y) <= threshold else 0.0

    return wrapped_func


def exponential(alpha: float = 1.0) -> model.DataSimilarityBatchFunc[Number]:
    @apply
    def wrapped_func(x: Number, y: Number) -> model.SimilarityValue:
        return math.exp(-alpha * abs(x - y))

    return wrapped_func


def sigmoid(
    alpha: float = 1.0, theta: float = 1.0
) -> model.DataSimilarityBatchFunc[Number]:
    @apply
    def wrapped_func(x: Number, y: Number) -> model.SimilarityValue:
        return 1.0 / (1.0 + math.exp((abs(x - y) - theta) / alpha))

    return wrapped_func