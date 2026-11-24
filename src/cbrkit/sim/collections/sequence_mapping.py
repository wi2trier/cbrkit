import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

from ...helpers import batchify
from ...model.similarity import SequenceSim
from ...schema import serializable
from ...typing import Float, SimFunc
from .._nested import NestedMeasure
from ._common import empty_mapping

__all__ = [
    "SequenceMapping",
    "Weight",
]


@dataclass(slots=True, frozen=True)
class Weight:
    """A weighted interval with bounds for sequence mapping similarity.

    Args:
        weight: Weight of the local similarities within the interval.
        lower_bound: Lower bound of the interval.
        upper_bound: Upper bound of the interval.
        inclusive_lower: Whether the lower bound belongs to the interval.
        inclusive_upper: Whether the upper bound belongs to the interval.
    """

    weight: float
    lower_bound: float
    upper_bound: float
    inclusive_lower: bool
    inclusive_upper: bool

    def contains(self, value: float) -> bool:
        """Whether the given similarity lies within the interval."""
        lower = (
            self.lower_bound <= value
            if self.inclusive_lower
            else self.lower_bound < value
        )
        upper = (
            value <= self.upper_bound
            if self.inclusive_upper
            else value < self.upper_bound
        )

        return lower and upper

    @property
    def normalized_weight(self) -> float:
        """The weight scaled by the width of the interval."""
        width = self.upper_bound - self.lower_bound

        return self.weight / width if width > 0.0 else self.weight


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SequenceMapping[V, S: Float](SimFunc[Sequence[V], SequenceSim[V, S]]):
    """Compares two sequences position by position, sliding the shorter one across the longer one.

    Every window of the longer sequence is compared with the shorter one element by element,
    and the window with the highest mean similarity is kept together with its mapping.
    With `exact`, sequences of different lengths have a similarity of zero instead.

    Args:
        element_measure: Similarity of two elements, called with the case element first.
        exact: Whether the sequences must have the same length instead of sliding.
        weights: Weighted intervals of the local similarities, which weight their mean.

    Examples:
        >>> sim = SequenceMapping(element_measure=lambda x, y: 1.0 if x == y else 0.0)
        >>> sim(["a", "b", "c"], ["a", "b", "c"])
        SequenceSim(value=1.0, similarities=(1.0, 1.0, 1.0),
            mapping=(('a', 'a'), ('b', 'b'), ('c', 'c')))
        >>> sim(["x", "a", "b"], ["a", "b"])
        SequenceSim(value=1.0, similarities=(1.0, 1.0), mapping=(('a', 'a'), ('b', 'b')))
        >>> SequenceMapping(element_measure=sim.element_measure, exact=True)(["x", "a", "b"], ["a", "b"]).value
        0.0
    """

    element_measure: NestedMeasure[V, S]
    exact: bool = False
    weights: Sequence[Weight] | None = None
    kind: Literal["sequence_mapping"] = field(default="sequence_mapping", repr=False)

    def aggregate(self, similarities: Sequence[S]) -> float:
        """Mean of the local similarities, weighted by `weights` if any of them applies."""
        values = [float(sim) for sim in similarities]
        weighted = [
            (weight.normalized_weight, value)
            for value in values
            for weight in self.weights or ()
            if weight.contains(value)
        ]
        total_weight = sum(weight for weight, _ in weighted)

        if total_weight > 0:
            return sum(weight * value for weight, value in weighted) / total_weight

        return statistics.fmean(values)

    @override
    def __call__(self, case: Sequence[V], query: Sequence[V]) -> SequenceSim[V, S]:
        if (empty := empty_mapping(case, query)) is not None:
            return empty

        if self.exact and len(case) != len(query):
            return SequenceSim(value=0.0, similarities=(), mapping=())

        # Pairs of case and query positions, one list per offset of the shorter sequence.
        windows = [
            [
                (start + i, i) if len(case) >= len(query) else (i, start + i)
                for i in range(min(len(case), len(query)))
            ]
            for start in range(abs(len(case) - len(query)) + 1)
        ]
        sims = iter(
            batchify(self.element_measure)(
                [(case[x], query[y]) for window in windows for x, y in window]
            )
        )
        local = [tuple(next(sims) for _ in window) for window in windows]
        values = [self.aggregate(similarities) for similarities in local]
        best = max(range(len(windows)), key=values.__getitem__)

        return SequenceSim(
            value=values[best],
            similarities=local[best],
            mapping=tuple((query[y], case[x]) for x, y in windows[best]),
        )
