from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from cbrkit.sim.helpers import sim2seq
from cbrkit.typing import (
    SimilaritySequence,
    SimilarityValue,
    SimSeqFunc,
    ValueType,
)


def table(
    *args: tuple[ValueType, ValueType, SimilarityValue],
    symmetric: bool = True,
    default: SimilarityValue = 0.0,
) -> SimSeqFunc[ValueType]:
    table: defaultdict[
        ValueType, defaultdict[ValueType, SimilarityValue]
    ] = defaultdict(lambda: defaultdict(lambda: default))

    for arg in args:
        table[arg[0]][arg[1]] = arg[2]

        if symmetric:
            table[arg[1]][arg[0]] = arg[2]

    def wrapped_func(
        pairs: Sequence[tuple[ValueType, ValueType]]
    ) -> SimilaritySequence:
        return [table[pair[0]][pair[1]] for pair in pairs]

    return wrapped_func


def equality() -> SimSeqFunc[Any]:
    @sim2seq
    def wrapped_func(x: Any, y: Any) -> SimilarityValue:
        return x == y

    return wrapped_func
