import statistics
from typing import Mapping, Sequence

from cbrkit import model

__all__ = ("equality", "aggregate", "get")


def equality(case: model.CaseType, query: model.CaseType) -> model.SimilarityValue:
    return case == query


_mapping: dict[model.SimilarityType, model.SimilarityFunc] = {
    "equality": equality,
}


def get(name: model.SimilarityType) -> model.SimilarityFunc[model.CaseType]:
    return _mapping[name]


def aggregate(
    operation: model.AggregationOperation,
    similarities: model.AggregationType,
    weights: model.AggregationType | None = None,
) -> model.SimilarityValue:
    assert weights is None or type(similarities) == type(weights)

    sims: Sequence[model.SimilarityValue]

    if isinstance(similarities, Mapping) and isinstance(weights, Mapping):
        sims = [sim * weights[key] for key, sim in similarities.items()]
    elif isinstance(similarities, Sequence) and isinstance(weights, Sequence):
        sims = [s * w for s, w in zip(similarities, weights, strict=True)]
    elif isinstance(similarities, Sequence) and weights is None:
        sims = similarities
    elif isinstance(similarities, Mapping) and weights is None:
        sims = list(similarities.values())
    else:
        raise NotImplementedError()

    if operation == "mean":
        return statistics.mean(sims)
    if operation == "fmean":
        return statistics.fmean(sims)
    if operation == "geometric_mean":
        return statistics.geometric_mean(sims)
    if operation == "harmonic_mean":
        return statistics.harmonic_mean(sims)
    elif operation == "median":
        return statistics.median(sims)
    elif operation == "median_low":
        return statistics.median_low(sims)
    elif operation == "median_high":
        return statistics.median_high(sims)
    elif operation == "mode":
        return statistics.mode(sims)
    elif operation == "min":
        return min(sims)
    elif operation == "max":
        return max(sims)
    elif operation == "sum":
        return sum(sims)
    else:
        raise NotImplementedError()
