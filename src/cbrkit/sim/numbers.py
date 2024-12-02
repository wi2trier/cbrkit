import math
from dataclasses import dataclass
from typing import override

from ..typing import SimFunc

type Number = float | int

__all__ = ["linear_interval", "linear", "threshold", "exponential", "sigmoid"]


@dataclass(slots=True, frozen=True)
class linear_interval(SimFunc[Number, float]):
    """Linear similarity function based on the distance between two values within a range.

    Args:
        min: Lower bound of the interval. Should be the minimal value of the entire case base.
        max: Upper bound of the interval. Should be the maximal value of the entire case base.

    Examples:
        >>> sim = linear_interval(1950, 2000)
        >>> sim(1950, 1975)
        0.5
    """

    min: float
    max: float

    @override
    def __call__(self, x: Number, y: Number) -> float:
        if x < self.min or x > self.max or y < self.min or y > self.max:
            return 0.0

        return 1.0 - abs(x - y) / (self.max - self.min)


@dataclass(slots=True, frozen=True)
class linear(SimFunc[Number, float]):
    """Linear similarity function based on the distance between two values.

    Args:
        max: Maximum bound of the distance (i.e., the point where the similarity is 0.0)
        min: Minimum bound of the distance (i.e., the point where the similarity is 1.0)

    ![linear](../../../assets/numeric/linear.png)

    Examples:
        >>> sim = linear(100)
        >>> sim(50, 60)
        0.9
    """

    max: float
    min: float = 0.0

    @override
    def __call__(self, x: Number, y: Number) -> float:
        dist = abs(x - y)

        if dist < self.min:
            return 1.0
        elif dist > self.max:
            return 0.0

        return (self.max - dist) / (self.max - self.min)


@dataclass(slots=True, frozen=True)
class threshold(SimFunc[Number, float]):
    """Threshold similarity function.

    Args:
        value: If the absolute difference between the two values is less than or equal to this value, the similarity is 1.0, otherwise it is 0.0

    ![threshold](../../../assets/numeric/threshold.png)

    Examples:
        >>> sim = threshold(10)
        >>> sim(50, 60)
        1.0
        >>> sim(50, 61)
        0.0
    """

    value: float

    @override
    def __call__(self, x: Number, y: Number) -> float:
        return 1.0 if abs(x - y) <= self.value else 0.0


@dataclass(slots=True, frozen=True)
class exponential(SimFunc[Number, float]):
    """Exponential similarity function.

    Args:
        alpha: Controls the growth of the exponential function for the similarity. The larger alpha is, the faster the similarity decreases.

    ![exponential](../../../assets/numeric/exponential.png)

    Examples:
        >>> sim = exponential(0.1)
        >>> sim(50, 60)
        0.36787944117144233
    """

    alpha: float = 1.0

    @override
    def __call__(self, x: Number, y: Number) -> float:
        return math.exp(-self.alpha * abs(x - y))


@dataclass(slots=True, frozen=True)
class sigmoid(SimFunc[Number, float]):
    """Sigmoid similarity function.

    Args:
        alpha: Specifies the steepness of the similarity decrease. The smaller alpha, the steeper is the decrease.
        theta: Specifies the point at which the similarity value is 0.5.

    ![sigmoid](../../../assets/numeric/sigmoid.png)

    Examples:
        >>> sim = sigmoid(1, 10)
        >>> sim(50, 60)
        0.5
        >>> sim(50, 58)
        0.8807970779778823
    """

    alpha: float = 1.0
    theta: float = 1.0

    @override
    def __call__(self, x: Number, y: Number) -> float:
        return 1.0 / (1.0 + math.exp((abs(x - y) - self.theta) / self.alpha))
