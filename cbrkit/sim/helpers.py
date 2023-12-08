import statistics
from collections.abc import Mapping, Sequence
from inspect import signature as inspect_signature
from typing import Any, Literal, cast

from cbrkit.typing import (
    AggregatorFunc,
    KeyType,
    SimFunc,
    SimMap,
    SimMapFunc,
    SimPairOrMapFunc,
    SimPairOrSeqFunc,
    SimSeq,
    SimSeqFunc,
    SimType,
    SimVals,
    ValueType,
)

__all__ = [
    "soft_sim2seq",
    "soft_sim2map",
    "dist2sim",
    "sim2seq",
    "sim2map",
    "aggregator",
]


def soft_sim2seq(func: SimPairOrSeqFunc[ValueType]) -> SimSeqFunc[ValueType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2:
        return sim2seq(cast(SimFunc[ValueType], func))

    return cast(SimSeqFunc[ValueType], func)


def soft_sim2map(
    func: SimPairOrMapFunc[KeyType, ValueType]
) -> SimMapFunc[KeyType, ValueType]:
    signature = inspect_signature(func)

    if signature.parameters.keys() == {"x", "y"}:
        return sim2map(cast(SimFunc[ValueType], func))

    return cast(SimMapFunc[KeyType, ValueType], func)


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


def sim2seq(func: SimFunc[ValueType]) -> SimSeqFunc[ValueType]:
    def wrapped_func(pairs: Sequence[tuple[ValueType, ValueType]]) -> SimSeq:
        return [func(x, y) for (x, y) in pairs]

    return wrapped_func


def sim2map(func: SimFunc[ValueType]) -> SimMapFunc[Any, ValueType]:
    def wrapped_func(
        x_map: Mapping[KeyType, ValueType],
        y: ValueType,
    ) -> SimMap[KeyType]:
        return {key: func(x, y) for key, x in x_map.items()}

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


def aggregator(
    pooling: Pooling = "mean",
    pooling_weights: SimVals[KeyType] | None = None,
    default_pooling_weight: float = 1.0,
) -> AggregatorFunc[KeyType]:
    def wrapped_func(similarities: SimVals[KeyType]) -> SimType:
        assert pooling_weights is None or type(similarities) == type(pooling_weights)

        sims: Sequence[SimType]  # noqa: F821

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
