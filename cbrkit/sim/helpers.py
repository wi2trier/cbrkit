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
    SimSeq,
    SimSeqFunc,
    SimSeqOrMap,
    SimVal,
    ValueType,
)

__all__ = [
    "dist2sim",
    "sim2seq",
    "sim2map",
    "aggregator",
]


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


def sim2seq(func: SimFunc[ValueType] | SimSeqFunc[ValueType]) -> SimSeqFunc[ValueType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2:
        casted_func = cast(SimFunc[ValueType], func)

        def wrapped_func(pairs: Sequence[tuple[ValueType, ValueType]]) -> SimSeq:
            return [casted_func(x, y) for (x, y) in pairs]

        return wrapped_func

    #     if len(signature.parameters) == 1:
    #         casted_func = cast(SimMapFunc[Any, ValueType], func)
    #         def wrapped_func(pairs: Sequence[tuple[ValueType, ValueType]]) -> SimSeq:
    #             pass
    #         return wrapped_func

    return cast(SimSeqFunc[ValueType], func)


def sim2map(
    func: SimFunc[ValueType] | SimSeqFunc[ValueType] | SimMapFunc[KeyType, ValueType],
) -> SimMapFunc[KeyType, ValueType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2 and signature.parameters.keys() == {"x", "y"}:
        sim_pair_func = cast(SimFunc[ValueType], func)

        def wrapped_sim_pair_func(
            x_map: Mapping[KeyType, ValueType],
            y: ValueType,
        ) -> SimMap[KeyType]:
            return {key: sim_pair_func(x, y) for key, x in x_map.items()}

        return wrapped_sim_pair_func

    elif len(signature.parameters) == 1:
        sim_seq_func = cast(SimSeqFunc[ValueType], func)

        def wrapped_sim_seq_func(
            x_map: Mapping[Any, ValueType], y: ValueType
        ) -> SimMap[KeyType]:
            pairs = [(x, y) for x in x_map.values()]
            sims = sim_seq_func(pairs)

            return {key: sim for key, sim in zip(x_map.keys(), sims, strict=True)}

        return wrapped_sim_seq_func

    return cast(SimMapFunc[KeyType, ValueType], func)


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
