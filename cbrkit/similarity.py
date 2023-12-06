import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cbrkit import model

__all__: tuple[str, ...] = (
    "equality",
    "aggregate",
    "get",
    "apply_global",
    "apply_local",
)


# TODO: Document that this shall be applied when using a CaseSimilaritySingleFunc
def apply_global(
    func: model.CaseSimilaritySingleFunc[model.CaseType],
) -> model.CaseSimilarityBatchFunc[model.CaseType]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType],
        query: model.CaseType,
    ) -> model.CaseSimilarityMap:
        return {key: func(case, query) for key, case in casebase.items()}

    return wrapped_func


# TODO: Document that this shall be applied when using a DataSimilaritySingleFunc
def apply_local(
    func: model.DataSimilaritySingleFunc[model.DataType]
) -> model.DataSimilarityBatchFunc[model.DataType]:
    def wrapped_func(
        *args: tuple[model.DataType, model.DataType]
    ) -> model.SimilaritySequence:
        return [func(data1, data2) for (data1, data2) in args]

    return wrapped_func


def by_attributes(
    mapping: Mapping[str, model.DataSimilarityBatchFunc],
    pooling: model.Pooling = "mean",
    pooling_weights: Mapping[str, float] | None = None,
    getter: Callable[[Any, str], Any] = getattr,
) -> model.CaseSimilarityBatchFunc:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

        for attr, sim_func in mapping.items():
            casebase_attribute_pairs = [
                (getter(case, attr), getter(query, attr)) for case in casebase.values()
            ]
            casebase_similarities = sim_func(*casebase_attribute_pairs)

            for casename, similarity in zip(
                casebase.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregate(similarities, pooling, pooling_weights)
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func


def equality(case: model.CaseType, query: model.CaseType) -> model.SimilarityValue:
    return case == query


_mapping: dict[model.SimilarityFuncName, model.CaseSimilarityBatchFunc] = {
    "equality": apply_global(equality),
}


def get(
    name: model.SimilarityFuncName
) -> model.CaseSimilarityBatchFunc[model.CaseType]:
    return _mapping[name]


def aggregate(
    similarities: model.SimilarityValues,
    pooling: model.Pooling,
    pooling_weights: model.SimilarityValues | None = None,
) -> model.SimilarityValue:
    assert pooling_weights is None or type(similarities) == type(pooling_weights)

    sims: Sequence[model.SimilarityValue]

    if isinstance(similarities, Mapping) and isinstance(pooling_weights, Mapping):
        sims = [sim * pooling_weights[key] for key, sim in similarities.items()]
    elif isinstance(similarities, Sequence) and isinstance(pooling_weights, Sequence):
        sims = [s * w for s, w in zip(similarities, pooling_weights, strict=True)]
    elif isinstance(similarities, Sequence) and pooling_weights is None:
        sims = similarities
    elif isinstance(similarities, Mapping) and pooling_weights is None:
        sims = list(similarities.values())
    else:
        raise NotImplementedError()

    if pooling == "mean":
        return statistics.mean(sims)
    if pooling == "fmean":
        return statistics.fmean(sims)
    if pooling == "geometric_mean":
        return statistics.geometric_mean(sims)
    if pooling == "harmonic_mean":
        return statistics.harmonic_mean(sims)
    elif pooling == "median":
        return statistics.median(sims)
    elif pooling == "median_low":
        return statistics.median_low(sims)
    elif pooling == "median_high":
        return statistics.median_high(sims)
    elif pooling == "mode":
        return statistics.mode(sims)
    elif pooling == "min":
        return min(sims)
    elif pooling == "max":
        return max(sims)
    elif pooling == "sum":
        return sum(sims)
    else:
        raise NotImplementedError()
