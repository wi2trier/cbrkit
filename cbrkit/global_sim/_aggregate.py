import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from cbrkit.typing import (
    AggregatorFunc,
    KeyType,
    SimSeqOrMap,
    SimVal,
)

__all__ = [
    "Pooling",
    "aggregator",
]


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


def aggregator(
    pooling: Pooling = "mean",
    pooling_weights: SimSeqOrMap[KeyType] | None = None,
    default_pooling_weight: float = 1.0,
) -> AggregatorFunc[KeyType]:
    def wrapped_func(similarities: SimSeqOrMap[KeyType]) -> SimVal:
        assert pooling_weights is None or type(similarities) == type(pooling_weights)

        sims: Sequence[SimVal]  # noqa: F821

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
