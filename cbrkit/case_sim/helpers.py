import statistics
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from cbrkit.typing import (
    AggregateFunc,
    Casebase,
    CaseName,
    CaseSimBatchFunc,
    CaseSimFunc,
    CaseType,
    SimilarityMap,
    SimilarityValue,
    SimilarityValues,
)


def batchify(
    func: CaseSimFunc[CaseType],
) -> CaseSimBatchFunc[Any, CaseType]:
    def wrapped_func(
        casebase: Casebase[CaseName, CaseType],
        query: CaseType,
    ) -> SimilarityMap[CaseName]:
        return {key: func(case, query) for key, case in casebase.items()}

    return wrapped_func


Pooling = Literal[
    "mean",
    "fmean",
    "geometric_mean",
    "harmonic_mean",
    "median",
    "median_low",
    "median_high",
    "mode",
    "min",
    "max",
    "sum",
]


def aggregate(
    pooling: Pooling = "mean",
    pooling_weights: SimilarityValues | None = None,
    default_pooling_weight: SimilarityValue = 1.0,
) -> AggregateFunc:
    def wrapped_func(similarities: SimilarityValues) -> SimilarityValue:
        assert pooling_weights is None or type(similarities) == type(pooling_weights)

        sims: Sequence[SimilarityValue]  # noqa: F821

        if isinstance(similarities, Mapping) and isinstance(pooling_weights, Mapping):
            sims = [
                sim * pooling_weights.get(key, default_pooling_weight)
                for key, sim in similarities.items()
            ]
        elif isinstance(similarities, Sequence) and isinstance(
            pooling_weights, Sequence
        ):
            sims = [s * w for s, w in zip(similarities, pooling_weights, strict=True)]
        elif isinstance(similarities, Sequence) and pooling_weights is None:
            sims = similarities
        elif isinstance(similarities, Mapping) and pooling_weights is None:
            sims = list(similarities.values())
        else:
            raise NotImplementedError()

        match pooling:
            case "mean":
                return statistics.mean(sims)
            case "fmean":
                return statistics.fmean(sims)
            case "geometric_mean":
                return statistics.geometric_mean(sims)
            case "harmonic_mean":
                return statistics.harmonic_mean(sims)
            case "median":
                return statistics.median(sims)
            case "median_low":
                return statistics.median_low(sims)
            case "median_high":
                return statistics.median_high(sims)
            case "mode":
                return statistics.mode(sims)
            case "min":
                return min(sims)
            case "max":
                return max(sims)
            case "sum":
                return sum(sims)
            case _:
                raise NotImplementedError()

    return wrapped_func


aggregate_default = aggregate()
