import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

import numpy as np
import scipy
from pydantic import Field

from ..schema import serializable
from ..typing import ColumnSimFunc, NumpyArray

type Number = float | int

__all__ = ["Exponential", "Linear", "LinearInterval", "Sigmoid", "Threshold"]


def _arrays(
    cases: Sequence[Number], queries: Sequence[Number]
) -> tuple[NumpyArray, NumpyArray]:
    return np.asarray(cases, dtype=np.float64), np.asarray(queries, dtype=np.float64)


def _distances(cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
    x, y = _arrays(cases, queries)

    return np.abs(x - y)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class LinearInterval(ColumnSimFunc[Number]):
    """Linear similarity function based on the distance between two values within a range.

    Args:
        lower: Lower bound of the interval. Should be the minimal value of the entire case base.
        upper: Upper bound of the interval. Should be the maximal value of the entire case base.

    Examples:
        >>> sim = LinearInterval(lower=1950, upper=2000)
        >>> sim(1950, 1975)
        0.5
    """

    lower: float
    upper: float
    kind: Literal["linear_interval"] = field(default="linear_interval", repr=False)

    def __post_init__(self) -> None:
        if self.upper < self.lower:
            raise ValueError("upper must not be smaller than lower")

    @override
    def __call__(self, case: Number, query: Number) -> float:
        if (
            case < self.lower
            or case > self.upper
            or query < self.lower
            or query > self.upper
        ):
            return 0.0

        if self.upper == self.lower:
            return 1.0

        return 1.0 - abs(case - query) / (self.upper - self.lower)

    @override
    def columns(self, cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
        x, y = _arrays(cases, queries)
        inside = (
            (x >= self.lower)
            & (x <= self.upper)
            & (y >= self.lower)
            & (y <= self.upper)
        )

        if self.upper == self.lower:
            return inside.astype(np.float64)

        return np.where(inside, 1.0 - np.abs(x - y) / (self.upper - self.lower), 0.0)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Linear(ColumnSimFunc[Number]):
    """Linear similarity function based on the distance between two values.

    Args:
        max_distance: Distance at which the similarity reaches 0.0.
        min_distance: Distance up to which the similarity is 1.0.

    ![linear](/assets/numeric/linear.png)

    Examples:
        >>> sim = Linear(max_distance=100)
        >>> sim(50, 60)
        0.9
    """

    max_distance: Annotated[float, Field(gt=0)]
    min_distance: Annotated[float, Field(ge=0)] = 0.0
    kind: Literal["linear"] = field(default="linear", repr=False)

    def __post_init__(self) -> None:
        if self.max_distance < self.min_distance:
            raise ValueError("max_distance must not be smaller than min_distance")

    @override
    def __call__(self, case: Number, query: Number) -> float:
        dist = abs(case - query)

        if dist <= self.min_distance:
            return 1.0

        if dist >= self.max_distance:
            return 0.0

        return (self.max_distance - dist) / (self.max_distance - self.min_distance)

    @override
    def columns(self, cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
        dist = _distances(cases, queries)

        if self.max_distance == self.min_distance:
            return (dist <= self.min_distance).astype(np.float64)

        return np.clip(
            (self.max_distance - dist) / (self.max_distance - self.min_distance),
            0.0,
            1.0,
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Threshold(ColumnSimFunc[Number]):
    """Threshold similarity function.

    Args:
        value: If the absolute difference between the two values is less than or equal to this value, the similarity is 1.0, otherwise it is 0.0

    ![threshold](/assets/numeric/threshold.png)

    Examples:
        >>> sim = Threshold(value=10)
        >>> sim(50, 60)
        1.0
        >>> sim(50, 61)
        0.0
    """

    value: Annotated[float, Field(ge=0)]
    kind: Literal["threshold"] = field(default="threshold", repr=False)

    @override
    def __call__(self, case: Number, query: Number) -> float:
        return 1.0 if abs(case - query) <= self.value else 0.0

    @override
    def columns(self, cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
        return (_distances(cases, queries) <= self.value).astype(np.float64)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Exponential(ColumnSimFunc[Number]):
    """Exponential similarity function.

    Args:
        alpha: Controls the growth of the exponential function for the similarity. The larger alpha is, the faster the similarity decreases.

    ![exponential](/assets/numeric/exponential.png)

    Examples:
        >>> sim = Exponential(alpha=0.1)
        >>> sim(50, 60)
        0.36787944117144233
    """

    alpha: Annotated[float, Field(ge=0)] = 1.0
    kind: Literal["exponential"] = field(default="exponential", repr=False)

    @override
    def __call__(self, case: Number, query: Number) -> float:
        return math.exp(-self.alpha * abs(case - query))

    @override
    def columns(self, cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
        return np.exp(-self.alpha * _distances(cases, queries))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Sigmoid(ColumnSimFunc[Number]):
    """Sigmoid similarity function.

    Args:
        alpha: Specifies the steepness of the similarity decrease. The smaller alpha, the steeper is the decrease.
        theta: Specifies the point at which the similarity value is 0.5.

    ![sigmoid](/assets/numeric/sigmoid.png)

    Examples:
        >>> sim = Sigmoid(alpha=1, theta=10)
        >>> sim(50, 60)
        0.5
        >>> sim(50, 58)
        0.8807970779778823
    """

    alpha: Annotated[float, Field(gt=0)] = 1.0
    theta: Annotated[float, Field(ge=0)] = 1.0
    kind: Literal["sigmoid"] = field(default="sigmoid", repr=False)

    @override
    def __call__(self, case: Number, query: Number) -> float:
        return float(
            scipy.special.expit(
                np.float64((self.theta - abs(case - query)) / self.alpha)
            )
        )

    @override
    def columns(self, cases: Sequence[Number], queries: Sequence[Number]) -> NumpyArray:
        return scipy.special.expit(
            (self.theta - _distances(cases, queries)) / self.alpha
        )
