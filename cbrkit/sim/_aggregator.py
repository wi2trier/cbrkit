import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from cbrkit.helpers import unpack_sim
from cbrkit.typing import (
    AggregatorFunc,
    AnyFloat,
    KeyType,
    PoolingFunc,
    SimSeqOrMap,
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
    pooling_weights: SimSeqOrMap[KeyType, float] | None = None,
    default_pooling_weight: float = 1.0,
) -> AggregatorFunc[KeyType, AnyFloat]:
    """
    Aggregates local similarities to a global similarity using the specified pooling function.

    Args:
        pooling: The pooling function to use. It can be either a string representing the name of the pooling function or a custom pooling function (see `cbrkit.typing.PoolingFunc`).
        pooling_weights: The weights to apply to the similarities during pooling. It can be a sequence or a mapping. If None, every local similarity is weighted equally.
        default_pooling_weight: The default weight to use if a similarity key is not found in the pooling_weights mapping.

    Examples:
        >>> agg = aggregator("mean")
        >>> agg([0.5, 0.75, 1.0])
        0.75
    """

    pooling_func = _pooling_funcs[pooling] if isinstance(pooling, str) else pooling

    def wrapped_func(similarities: SimSeqOrMap[KeyType, AnyFloat]) -> float:
        assert pooling_weights is None or type(similarities) == type(pooling_weights)

        sims: Sequence[float]  # noqa: F821

        if isinstance(similarities, Mapping) and isinstance(pooling_weights, Mapping):
            sims = [
                unpack_sim(sim) * pooling_weights.get(key, default_pooling_weight)
                for key, sim in similarities.items()
            ]
        elif isinstance(similarities, Sequence) and isinstance(
            pooling_weights, Sequence
        ):
            sims = [
                unpack_sim(s) * w
                for s, w in zip(similarities, pooling_weights, strict=True)
            ]
        elif isinstance(similarities, Sequence) and pooling_weights is None:
            sims = [unpack_sim(s) for s in similarities]
        elif isinstance(similarities, Mapping) and pooling_weights is None:
            sims = [unpack_sim(s) for s in similarities.values()]
        else:
            raise NotImplementedError()

        return pooling_func(sims)

    return wrapped_func
