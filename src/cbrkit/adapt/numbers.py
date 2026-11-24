from dataclasses import dataclass, field
from typing import Annotated, Literal, override

from ..schema import serializable
from ..sim.pooling import POOLINGS, Mean
from ..typing import AdaptationFunc, PoolingFunc

type Number = float | int

__all__ = [
    "Aggregate",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Aggregate(AdaptationFunc[Number]):
    """Aggregate two numbers using a pooling function.

    Args:
        pooling: Pooling function of the two values, such as `cbrkit.sim.pooling.Mean`.
        case_factor: Factor to multiply the case value by.
        query_factor: Factor to multiply the query value by.

    Examples:
        >>> func1 = Aggregate()
        >>> func1(10, 20)
        15.0
        >>> func2 = Aggregate(case_factor=3, query_factor=0.5)
        >>> func2(10, 20)
        20.0
    """

    pooling: Annotated[PoolingFunc[float], POOLINGS] = field(default_factory=Mean)
    case_factor: float = 1.0
    query_factor: float = 1.0
    kind: Literal["aggregate"] = field(default="aggregate", repr=False)

    @override
    def __call__(self, case: Number, query: Number) -> float:
        return self.pooling([case * self.case_factor, query * self.query_factor])
