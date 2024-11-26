import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, override

from ..helpers import unpack_sim
from ..typing import (
    AggregatorFunc,
    Float,
    PoolingFunc,
    SimSeqOrMap,
)

__all__ = [
    "PoolingName",
    "pooling_funcs",
    "aggregator",
]


type PoolingName = Literal[
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

pooling_funcs: dict[PoolingName, PoolingFunc] = {
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


@dataclass(slots=True, frozen=True)
class aggregator[K](AggregatorFunc[K, Float]):
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
        >>> agg = aggregator("mean", {1: 1, 2: 1, 3: 0})
        >>> agg({1: 1, 2: 1, 3: 1})
        1.0
        >>> agg = aggregator("mean", {1: 1, 2: 1, 3: 2})
        >>> agg({1: 1, 2: 1, 3: 1})
        1.0
    """

    pooling: PoolingName | PoolingFunc = "mean"
    pooling_weights: SimSeqOrMap[K, float] | None = None
    default_pooling_weight: float = 1.0

    @override
    def __call__(self, similarities: SimSeqOrMap[K, Float]) -> float:
        pooling_func = (
            pooling_funcs[self.pooling]
            if isinstance(self.pooling, str)
            else self.pooling
        )
        assert (self.pooling_weights is None) or (
            type(similarities) is type(self.pooling_weights)  # noqa: E721
        )

        pooling_factor = 1.0
        sims: Sequence[float]  # noqa: F821

        if isinstance(similarities, Mapping) and isinstance(
            self.pooling_weights, Mapping
        ):
            sims = [
                unpack_sim(sim)
                * self.pooling_weights.get(key, self.default_pooling_weight)
                for key, sim in similarities.items()
            ]
            pooling_factor = len(similarities) / sum(
                self.pooling_weights.get(key, self.default_pooling_weight)
                for key in similarities.keys()
            )
        elif isinstance(similarities, Sequence) and isinstance(
            self.pooling_weights, Sequence
        ):
            sims = [
                unpack_sim(s) * w
                for s, w in zip(similarities, self.pooling_weights, strict=True)
            ]
            pooling_factor = len(similarities) / sum(self.pooling_weights)
        elif isinstance(similarities, Sequence) and self.pooling_weights is None:
            sims = [unpack_sim(s) for s in similarities]
        elif isinstance(similarities, Mapping) and self.pooling_weights is None:
            sims = [unpack_sim(s) for s in similarities.values()]
        else:
            raise NotImplementedError()

        return pooling_func(sims) * pooling_factor
