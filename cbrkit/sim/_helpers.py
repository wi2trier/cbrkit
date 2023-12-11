from collections.abc import Mapping, Sequence
from inspect import signature as inspect_signature
from typing import Any, cast

from cbrkit.typing import (
    KeyType,
    SimMap,
    SimMapFunc,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
    ValueType,
)

__all__ = [
    "dist2sim",
    "sim2seq",
    "sim2map",
]


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


def sim2seq(
    func: SimPairFunc[ValueType] | SimSeqFunc[ValueType],
) -> SimSeqFunc[ValueType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2:
        casted_func = cast(SimPairFunc[ValueType], func)

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
    func: SimPairFunc[ValueType]
    | SimSeqFunc[ValueType]
    | SimMapFunc[KeyType, ValueType],
) -> SimMapFunc[KeyType, ValueType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2 and signature.parameters.keys() == {"x", "y"}:
        sim_pair_func = cast(SimPairFunc[ValueType], func)

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
