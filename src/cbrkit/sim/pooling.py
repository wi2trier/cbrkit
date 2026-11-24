"""Pooling functions that combine several values, such as local similarities, into one.

Every pooling function is a frozen class of a kind of `POOLINGS`, also those without parameters.
Most also have a weighted variant (`cbrkit.typing.WeightedPoolingFunc`) that `cbrkit.sim.Aggregator` uses,
for example the weighted arithmetic mean or the weighted minimum and maximum from fuzzy logic,
and a numpy variant for whole matrices (`cbrkit.typing.ColumnPoolingFunc`).
"""

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

import numpy as np
import numpy.typing as npt
import scipy
from pydantic import Field

from ..schema import Family, serializable
from ..typing import ColumnPoolingFunc, NumpyArray, PoolingFunc, WeightedPoolingFunc

__all__ = [
    "GeometricMean",
    "HarmonicMean",
    "KMax",
    "KMin",
    "Max",
    "Mean",
    "Median",
    "MedianHigh",
    "MedianLow",
    "Min",
    "Minkowski",
    "Mode",
    "Sum",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Mean(ColumnPoolingFunc):
    """The arithmetic mean.

    Examples:
        >>> Mean()([0.5, 1.0]), Mean().weighted([0.5, 1.0], [1.5, 0.5])
        (0.75, 0.625)
    """

    kind: Literal["mean"] = field(default="mean", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.fmean(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return statistics.fmean(values, weights)

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        return (values * weights).sum(axis=1) / weights.sum(axis=1)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class GeometricMean(ColumnPoolingFunc):
    """The geometric mean, 0 if a value is not positive.

    Examples:
        >>> GeometricMean()([0.25, 1.0])
        0.5
    """

    kind: Literal["geometric_mean"] = field(default="geometric_mean", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return self.weighted(values, [1.0] * len(values))

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        if min(values) <= 0:
            return 0.0

        return float(scipy.stats.gmean(values, weights=weights))

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        logs = np.log(np.where(kept, values, 1.0))
        zero = (kept & (values <= 0)).any(axis=1)

        return np.where(
            zero, 0.0, np.exp((weights * logs).sum(axis=1) / weights.sum(axis=1))
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class HarmonicMean(ColumnPoolingFunc):
    """The harmonic mean, 0 if a value is 0.

    Examples:
        >>> HarmonicMean()([0.5, 1.0])
        0.666...
    """

    kind: Literal["harmonic_mean"] = field(default="harmonic_mean", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.harmonic_mean(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return statistics.harmonic_mean(values, weights)

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        zero = (kept & (values == 0)).any(axis=1)
        inverse = weights / np.where(kept, values, 1.0)

        return np.where(zero, 0.0, weights.sum(axis=1) / inverse.sum(axis=1))


def _weighted_medians(
    values: Sequence[float], weights: Sequence[float]
) -> tuple[float, float]:
    """Return the low and high weighted median."""
    half = sum(weights) / 2
    cumulative = 0.0
    low: float | None = None

    for value, weight in sorted(zip(values, weights, strict=True)):
        cumulative += weight
        at_half = math.isclose(cumulative, half)

        if low is None and (cumulative > half or at_half):
            low = value

        if cumulative > half and not at_half:
            return low if low is not None else value, value

    raise ValueError("Weights must be positive")


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Median(WeightedPoolingFunc):
    """The median, the mean of the two middle values for an even number of values.

    Examples:
        >>> Median()([0.1, 0.5, 1.0, 0.2])
        0.35
    """

    kind: Literal["median"] = field(default="median", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.median(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return statistics.fmean(_weighted_medians(values, weights))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class MedianLow(WeightedPoolingFunc):
    """The low median, the smaller of the two middle values for an even number of values.

    Examples:
        >>> MedianLow()([0.1, 0.5, 1.0, 0.2])
        0.2
    """

    kind: Literal["median_low"] = field(default="median_low", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.median_low(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return _weighted_medians(values, weights)[0]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class MedianHigh(WeightedPoolingFunc):
    """The high median, the larger of the two middle values for an even number of values.

    Examples:
        >>> MedianHigh()([0.1, 0.5, 1.0, 0.2])
        0.5
    """

    kind: Literal["median_high"] = field(default="median_high", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.median_high(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return _weighted_medians(values, weights)[1]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Mode(WeightedPoolingFunc):
    """The most common value, the one with the largest total weight if weighted.

    Examples:
        >>> Mode()([0.5, 0.5, 1.0])
        0.5
    """

    kind: Literal["mode"] = field(default="mode", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return statistics.mode(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        totals: dict[float, float] = {}

        for value, weight in zip(values, weights, strict=True):
            totals[value] = totals.get(value, 0.0) + weight

        return max(totals, key=lambda value: totals[value])


# weighted minimum and maximum as known from fuzzy logic (Dubois and Prade),
# a weight of 1 keeps the value while a weight of 0 neutralizes it
@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Min(ColumnPoolingFunc):
    """The minimum, the weighted minimum of fuzzy logic if weighted.

    Examples:
        >>> Min()([0.5, 1.0]), Min().weighted([0.5, 1.0], [0.5, 1.5])
        (0.5, 0.666...)
    """

    kind: Literal["min"] = field(default="min", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return min(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        top = max(weights)

        return min(max(1 - w / top, v) for v, w in zip(values, weights, strict=True))

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        top = weights.max(axis=1, keepdims=True)
        terms = np.maximum(1 - weights / top, values)

        return np.where(kept, terms, np.inf).min(axis=1)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Max(ColumnPoolingFunc):
    """The maximum, the weighted maximum of fuzzy logic if weighted.

    Examples:
        >>> Max()([0.5, 1.0]), Max().weighted([0.5, 1.0], [1.5, 0.5])
        (1.0, 0.5)
    """

    kind: Literal["max"] = field(default="max", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return max(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        top = max(weights)

        return max(min(w / top, v) for v, w in zip(values, weights, strict=True))

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        top = weights.max(axis=1, keepdims=True)
        terms = np.minimum(weights / top, values)

        return np.where(kept, terms, -np.inf).max(axis=1)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Sum(ColumnPoolingFunc):
    """The sum, of the values multiplied by their weights if weighted.

    Examples:
        >>> Sum()([0.5, 1.0])
        1.5
    """

    kind: Literal["sum"] = field(default="sum", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return math.fsum(values)

    @override
    def weighted(self, values: Sequence[float], weights: Sequence[float]) -> float:
        return math.fsum(v * w for v, w in zip(values, weights, strict=True))

    @override
    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_]
    ) -> NumpyArray:
        return (values * weights).sum(axis=1)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class KMin(PoolingFunc[float]):
    """The k-th smallest value, or the largest one if there are fewer values.

    Args:
        k: Rank of the value, 1 for the minimum.

    Examples:
        >>> KMin(k=2)([0.5, 0.1, 0.9])
        0.5
    """

    k: Annotated[int, Field(ge=1)] = 1
    kind: Literal["k_min"] = field(default="k_min", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        ranked = sorted(values)

        return ranked[min(self.k, len(ranked)) - 1]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class KMax(PoolingFunc[float]):
    """The k-th largest value, or the smallest one if there are fewer values.

    Args:
        k: Rank of the value, 1 for the maximum.

    Examples:
        >>> KMax(k=2)([0.5, 0.1, 0.9])
        0.5
    """

    k: Annotated[int, Field(ge=1)] = 1
    kind: Literal["k_max"] = field(default="k_max", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        ranked = sorted(values, reverse=True)

        return ranked[min(self.k, len(ranked)) - 1]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Minkowski(PoolingFunc[float]):
    """The Minkowski norm (p-norm) of the values, the Euclidean norm by default.

    Args:
        p: Order of the norm, infinity gives the maximum and negative infinity the minimum.

    Examples:
        >>> Minkowski()([3.0, 4.0])
        5.0
    """

    p: float = 2.0
    kind: Literal["minkowski"] = field(default="minkowski", repr=False)

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return float(np.linalg.norm(values, ord=self.p))


POOLINGS = Family(
    "pooling",
    (
        Mean,
        GeometricMean,
        HarmonicMean,
        Median,
        MedianLow,
        MedianHigh,
        Mode,
        Min,
        Max,
        Sum,
        KMin,
        KMax,
        Minkowski,
    ),
)
