from abc import ABC
from collections.abc import Iterable, Mapping, Sequence
from inspect import signature as inspect_signature
from typing import Any, cast

from cbrkit.typing import (
    AnyFloat,
    KeyType,
    SimMap,
    SimMapFunc,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
    SimType,
    ValueType,
)

__all__ = [
    "dist2sim",
    "sim2seq",
    "sim2map",
    "unpack_sim",
    "unpack_sims",
    "AbstractFloat",
]


def dist2sim(distance: float) -> float:
    return 1 / (1 + distance)


def sim2seq(
    func: SimPairFunc[ValueType, SimType] | SimSeqFunc[ValueType, SimType],
) -> SimSeqFunc[ValueType, SimType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2:
        casted_func = cast(SimPairFunc[ValueType, SimType], func)

        def wrapped_func(pairs: Sequence[tuple[ValueType, ValueType]]) -> SimSeq:
            return [casted_func(x, y) for (x, y) in pairs]

        return wrapped_func

    return cast(SimSeqFunc[ValueType, SimType], func)


def sim2map(
    func: SimPairFunc[ValueType, SimType]
    | SimSeqFunc[ValueType, SimType]
    | SimMapFunc[KeyType, ValueType, SimType],
) -> SimMapFunc[KeyType, ValueType, SimType]:
    signature = inspect_signature(func)

    if len(signature.parameters) == 2 and signature.parameters.keys() == {"x", "y"}:
        sim_pair_func = cast(SimPairFunc[ValueType, SimType], func)

        def wrapped_sim_pair_func(
            x_map: Mapping[KeyType, ValueType],
            y: ValueType,
        ) -> SimMap[KeyType, SimType]:
            return {key: sim_pair_func(x, y) for key, x in x_map.items()}

        return wrapped_sim_pair_func

    elif len(signature.parameters) == 1:
        sim_seq_func = cast(SimSeqFunc[ValueType, SimType], func)

        def wrapped_sim_seq_func(
            x_map: Mapping[Any, ValueType], y: ValueType
        ) -> SimMap[KeyType, SimType]:
            pairs = [(x, y) for x in x_map.values()]
            sims = sim_seq_func(pairs)

            return {key: sim for key, sim in zip(x_map.keys(), sims, strict=True)}

        return wrapped_sim_seq_func

    return cast(SimMapFunc[KeyType, ValueType, SimType], func)


def unpack_sim(sim: AnyFloat) -> float:
    if isinstance(sim, float | int | bool):
        return sim
    else:
        return sim.value


def unpack_sims(sims: Iterable[AnyFloat]) -> list[float]:
    return [unpack_sim(sim) for sim in sims]


class AbstractFloat(ABC, float):
    def __new__(cls, *args, **kwargs):
        return float.__new__(cls, args[0])
