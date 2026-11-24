from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

import numpy as np
import numpy.typing as npt

from ..schema import serializable
from ..typing import (
    ColumnPoolingFunc,
    Float,
    NumpyArray,
    PoolingFunc,
    SimMap,
    SimSeq,
    WeightedPoolingFunc,
)
from .pooling import POOLINGS, Mean

__all__ = [
    "Aggregator",
    "default_aggregator",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Aggregator[K = str]:
    """
    Aggregates local similarities to a global similarity using the specified pooling function.

    Similarities with a weight of zero (or below) are left out.
    Pooling functions with a weighted variant (`cbrkit.typing.WeightedPoolingFunc`) use it,
    for example the weighted arithmetic mean or the weighted minimum and maximum from fuzzy logic.
    Other pooling functions receive the similarities multiplied by the weights normalized to a mean of 1.
    If no similarity is left, the result is 0.0.

    Args:
        pooling: Pooling function of the similarities, such as `cbrkit.sim.pooling.Mean`.
        weights: Weights of the similarities by key, or by position for sequences of similarities,
            `default_weight` for the others.
        default_weight: Weight of a similarity whose key has no weight.

    Examples:
        >>> from cbrkit.sim.pooling import Min
        >>> agg = Aggregator()
        >>> agg([0.5, 0.75, 1.0])
        0.75
        >>> agg = Aggregator(weights={1: 1, 2: 1, 3: 0})
        >>> agg({1: 1, 2: 1, 3: 1})
        1.0
        >>> agg = Aggregator(weights={1: 1, 2: 1, 3: 2})
        >>> agg({1: 1, 2: 1, 3: 1})
        1.0
        >>> agg = Aggregator(pooling=Min(), weights={1: 1, 2: 0.5})
        >>> agg({1: 1, 2: 0})
        0.5
    """

    pooling: Annotated[PoolingFunc[float], POOLINGS] = field(default_factory=Mean)
    weights: Mapping[K, float] = field(default_factory=dict)
    default_weight: float = 1.0

    def weight(self, key: K) -> float:
        """The weight of the similarity of a key, `default_weight` if it has none."""
        return self.weights.get(key, self.default_weight)

    def _weights(self, similarities: SimMap[K, Float] | SimSeq[Float]) -> list[float]:
        if isinstance(similarities, Mapping):
            return [self.weight(key) for key in similarities]

        # sequences are weighted by position
        weights: Mapping[Any, float] = self.weights

        return [
            weights.get(index, self.default_weight)
            for index in range(len(similarities))
        ]

    def __call__(self, similarities: SimMap[K, Float] | SimSeq[Float]) -> float:
        values = (
            similarities.values()
            if isinstance(similarities, Mapping)
            else cast(SimSeq[Float], similarities)
        )
        pairs = [
            (float(sim), weight)
            for sim, weight in zip(values, self._weights(similarities), strict=True)
            if weight > 0
        ]

        if not pairs:
            return 0.0

        sims = [sim for sim, _ in pairs]
        factor = len(pairs) / sum(weight for _, weight in pairs)
        weights = [weight * factor for _, weight in pairs]

        if not self.weights:
            return self.pooling(sims)

        if isinstance(self.pooling, WeightedPoolingFunc):
            return self.pooling.weighted(sims, weights)

        return self.pooling([s * w for s, w in zip(sims, weights, strict=True)])

    def columns(
        self,
        keys: Sequence[K],
        similarities: NumpyArray,
        present: npt.NDArray[np.bool_],
    ) -> NumpyArray:
        """Aggregates every row of a similarity matrix like a mapping of its present similarities.

        Pooling functions with a numpy variant (`cbrkit.typing.ColumnPoolingFunc`) pool all rows at once,
        other pooling functions and weights by position aggregate row by row.

        Args:
            keys: The keys of the columns.
            similarities: The similarities with one row per mapping and one column per key.
            present: Whether a similarity is part of its row.

        Examples:
            >>> agg = Aggregator(weights={"a": 3, "b": 1})
            >>> sims = np.array([[1.0, 0.0], [0.5, 0.0]])
            >>> agg.columns(["a", "b"], sims, np.array([[True, True], [True, False]]))
            array([0.75, 0.5 ])
        """
        if not isinstance(self.pooling, ColumnPoolingFunc):
            rows = zip(similarities.tolist(), present.tolist(), strict=True)

            return np.array(
                [
                    self({k: v for k, v, p in zip(keys, row, mask, strict=True) if p})
                    for row, mask in rows
                ],
                dtype=np.float64,
            )

        weights = np.array([self.weight(key) for key in keys], dtype=np.float64)
        kept = present & (weights > 0)
        count = kept.sum(axis=1)
        values = np.where(kept, similarities, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(kept, weights, 0.0)
            normalized *= (count / normalized.sum(axis=1))[:, None]
            result = self.pooling.columns(values, normalized, kept)

        return np.where(count > 0, result, 0.0)


default_aggregator = Aggregator()
