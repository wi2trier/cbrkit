from dataclasses import dataclass
from typing import override

from cbrkit.sim._aggregator import PoolingName, pooling_funcs
from cbrkit.typing import AdaptPairFunc, PoolingFunc, SupportsMetadata

type Number = float | int

__all__ = [
    "aggregate",
]


@dataclass(slots=True)
class aggregate(AdaptPairFunc[Number], SupportsMetadata):
    pooling: PoolingName | PoolingFunc = "mean"
    case_factor: Number = 1.0
    query_factor: Number = 1.0

    @override
    def __call__(self, case: Number, query: Number) -> float:
        pooling_func = (
            pooling_funcs[self.pooling]
            if isinstance(self.pooling, str)
            else self.pooling
        )

        values = [
            case * self.case_factor,
            query * self.query_factor,
        ]

        return pooling_func(values)
