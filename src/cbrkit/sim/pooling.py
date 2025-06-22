"""Custom pooling functions for similarity aggregation."""

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, override

from ..typing import PoolingFunc

__all__ = [
    "PoolingName",
    "pooling_funcs",
    "k_min",
    "k_max",
    "minkowski",
    "euclidean",
]


type PoolingName = Literal[
    "mean",
    "fmean",
    "geometric_mean",
    "harmonic_mean",
    "median",
    "median_low",
    "median_high",
    "mode",
    "min",
    "max",
    "sum",
]

pooling_funcs: dict[PoolingName, PoolingFunc[float]] = {
    "mean": statistics.mean,
    "fmean": statistics.fmean,
    "geometric_mean": statistics.geometric_mean,
    "harmonic_mean": statistics.harmonic_mean,
    "median": statistics.median,
    "median_low": statistics.median_low,
    "median_high": statistics.median_high,
    "mode": statistics.mode,
    "min": min,
    "max": max,
    "sum": sum,
}


@dataclass(slots=True, frozen=True)
class k_min(PoolingFunc[float]):
    """Return the k-th smallest value (1-indexed)."""

    k: int = 1

    @override
    def __call__(self, values: Sequence[float]) -> float:
        # Handle k=0 as k=1
        k = max(1, self.k)
        sorted_values = sorted(values)
        return sorted_values[min(k - 1, len(sorted_values) - 1)]


@dataclass(slots=True, frozen=True)
class k_max(PoolingFunc[float]):
    """Return the k-th largest value (1-indexed)."""

    k: int = 1

    @override
    def __call__(self, values: Sequence[float]) -> float:
        # Handle k=0 as k=1
        k = max(1, self.k)
        sorted_values = sorted(values, reverse=True)
        return sorted_values[min(k - 1, len(sorted_values) - 1)]


@dataclass(slots=True, frozen=True)
class minkowski(PoolingFunc[float]):
    """Compute the Minkowski norm (p-norm) of values."""

    p: float = 2.0

    @override
    def __call__(self, values: Sequence[float]) -> float:
        if self.p == float("inf"):
            return max(values)
        elif self.p == float("-inf"):
            return min(values)
        else:
            # Note: For weighted Minkowski in aggregator, weights are applied before calling this function
            return sum(abs(v) ** self.p for v in values) ** (1 / self.p)


@dataclass(slots=True, frozen=True)
class euclidean(PoolingFunc[float]):
    """Compute the Euclidean norm (2-norm) of values."""

    @override
    def __call__(self, values: Sequence[float]) -> float:
        return sum(v**2 for v in values) ** 0.5
