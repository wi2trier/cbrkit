import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from cbrkit.typing import (
    AggregatorFunc,
    KeyType,
    PoolingFunc,
    SimSeqOrMap,
    SimVal,
)

__all__ = [
    "PoolingName",
    "aggregator",
]


PoolingName = Literal[
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

_pooling_funcs: dict[PoolingName, PoolingFunc] = {
    "mean": statistics.mean,
    "fmean": statistics.fmean,
    "geometric_mean": statistics.geometric_mean,
    "harmonic_mean": statistics.harmonic_mean,
    "median": statistics.median,
    "median_low": statistics.median_low,
    "median_high": statistics.median_high,
    "mode": statistics.mode,
    "min": min,
    "max": max,
    "sum": sum,
}


def aggregator(
    pooling: PoolingName | PoolingFunc = "mean",
    pooling_weights: SimSeqOrMap[KeyType] | None = None,
    default_pooling_weight: float = 1.0,
) -> AggregatorFunc[KeyType]:
    pooling_func = _pooling_funcs[pooling] if isinstance(pooling, str) else pooling

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

        return pooling_func(sims)

    return wrapped_func
