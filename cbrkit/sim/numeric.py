import math

from cbrkit.typing import SimPairFunc

Number = float | int

__all__ = ["linear", "threshold", "exponential", "sigmoid"]


def linear(max: float, min: float = 0.0) -> SimPairFunc[Number, float]:
    """Linear similarity function.

    Args:
        max: Maximum bound of the interval
        min: Minimum bound of the interval

    ![linear](../../assets/numeric/linear.png)
    """

    def wrapped_func(x: Number, y: Number) -> float:
        dist = abs(x - y)

        if dist < min:
            return 1.0
        elif dist > max:
            return 0.0

        return (max - dist) / (max - min)

    return wrapped_func


def threshold(threshold: float) -> SimPairFunc[Number, float]:
    """Threshold similarity function.

    Args:
        threshold: If the absolute difference between the two values is less than or equal to this value, the similarity is 1.0, otherwise it is 0.0

    ![threshold](../../assets/numeric/threshold.png)
    """

    def wrapped_func(x: Number, y: Number) -> float:
        return 1.0 if abs(x - y) <= threshold else 0.0

    return wrapped_func


def exponential(alpha: float = 1.0) -> SimPairFunc[Number, float]:
    """Exponential similarity function.

    Args:
        alpha: Controls the growth of the exponential function for the similarity. The larger alpha is, the faster the function grows.

    ![exponential](../../assets/numeric/exponential.png)
    """

    def wrapped_func(x: Number, y: Number) -> float:
        return math.exp(-alpha * abs(x - y))

    return wrapped_func


def sigmoid(alpha: float = 1.0, theta: float = 1.0) -> SimPairFunc[Number, float]:
    """Sigmoid similarity function.

    Args:
        alpha: Specifies the steepness of the similarity decrease. The smaller alpha, the steeper is the decrease.
        theta: Specifies the point at which the similarity value is 0.5.

    ![sigmoid](../../assets/numeric/sigmoid.png)
    """

    def wrapped_func(x: Number, y: Number) -> float:
        return 1.0 / (1.0 + math.exp((abs(x - y) - theta) / alpha))

    return wrapped_func
