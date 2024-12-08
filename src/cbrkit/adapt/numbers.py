from dataclasses import dataclass
from typing import override

from ..sim.aggregator import PoolingName, pooling_funcs
from ..typing import AdaptationFunc, PoolingFunc

type Number = float | int

__all__ = [
    "aggregate",
]


@dataclass(slots=True)
class aggregate(AdaptationFunc[Number]):
    """Aggregate two numbers using a pooling function.

    Args:
        pooling: Pooling function or name of a pooling function.
        case_factor: Factor to multiply the case value by.
        query_factor: Factor to multiply the query value by.

    Returns:
        The aggregated value.

    Examples:
        >>> func1 = aggregate("mean")
        >>> func1(10, 20)
        15.0
        >>> func2 = aggregate("mean", case_factor=3, query_factor=0.5)
        >>> func2(10, 20)
        20.0
    """

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
