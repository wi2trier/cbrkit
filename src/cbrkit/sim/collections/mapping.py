import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import scipy

from ...model.similarity import SequenceSim
from ...schema import serializable
from ...typing import Float, SimFunc
from .._nested import NestedMeasure
from ._common import empty_mapping, similarity_matrix

__all__ = [
    "IsolatedMapping",
    "OptimalMapping",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class IsolatedMapping[V, S: Float](SimFunc[Sequence[V], SequenceSim[V, S]]):
    """Maps every query element to its most similar case element and averages their similarities.

    Case elements may be the best match of several query elements.
    The result keeps the mapping and the similarity of every query element as returned by the element measure.

    Args:
        element_measure: Similarity of two elements, called with the case element first.

    Examples:
        >>> from cbrkit.sim.strings import Levenshtein
        >>> sim = IsolatedMapping(element_measure=Levenshtein())
        >>> result = sim(["kitten", "sitting"], ["sitting", "fitted"])
        >>> result.value, result.mapping
        (0.833..., (('sitting', 'sitting'), ('fitted', 'kitten')))
        >>> sim([], [])
        SequenceSim(value=1.0, similarities=(), mapping=())
    """

    element_measure: NestedMeasure[V, S]
    kind: Literal["isolated_mapping"] = field(default="isolated_mapping", repr=False)

    @override
    def __call__(self, case: Sequence[V], query: Sequence[V]) -> SequenceSim[V, S]:
        if (empty := empty_mapping(case, query)) is not None:
            return empty

        sims = similarity_matrix(self.element_measure, case, query)
        best: list[int] = sims.floats.argmax(axis=1).tolist()

        return SequenceSim(
            value=statistics.fmean(
                float(sims.floats[row, col]) for row, col in enumerate(best)
            ),
            similarities=tuple(sims[row, col] for row, col in enumerate(best)),
            mapping=tuple(
                (item, case[col]) for item, col in zip(query, best, strict=True)
            ),
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class OptimalMapping[V, S: Float](SimFunc[Sequence[V], SequenceSim[V, float | S]]):
    """
    Computes the best possible mapping of the query items onto the case items.

    Each query item is mapped to at most one case item and each case item is used at
    most once, which is a maximum weight bipartite matching and is solved exactly.
    If the query is longer than the case, the surplus query items remain unmapped and
    contribute a similarity of zero.
    The result keeps the mapping and the similarity of every query item as returned by the element measure,
    an unmapped one is paired with None and a similarity of zero.

    Args:
        element_measure: Similarity of two elements, called with the case element first.

    Examples:
        >>> def example_similarity_function(x, y) -> float:
        ...     return 1.0 if x == y else 0.0
        >>> measure = OptimalMapping(element_measure=example_similarity_function)
        >>> measure(["Monday", "Tuesday", "Wednesday"], ["Monday", "Tuesday", "Sunday"]).value
        0.666...
        >>> result = measure(["Monday"], ["Monday", "Tuesday", "Wednesday"])
        >>> result.value, result.mapping
        (0.333..., (('Monday', 'Monday'), ('Tuesday', None), ('Wednesday', None)))
    """

    element_measure: NestedMeasure[V, S]
    kind: Literal["optimal_mapping"] = field(default="optimal_mapping", repr=False)

    @override
    def __call__(
        self, case: Sequence[V], query: Sequence[V]
    ) -> SequenceSim[V, float | S]:
        if (empty := empty_mapping(case, query)) is not None:
            return empty

        sims = similarity_matrix(self.element_measure, case, query)
        rows, cols = scipy.optimize.linear_sum_assignment(sims.floats, maximize=True)
        assigned: dict[int, int] = dict(zip(rows.tolist(), cols.tolist(), strict=True))

        return SequenceSim(
            value=sum(float(sims.floats[row, col]) for row, col in assigned.items())
            / len(query),
            similarities=tuple(
                sims[row, assigned[row]] if row in assigned else 0.0
                for row in range(len(query))
            ),
            mapping=tuple(
                (item, case[assigned[row]] if row in assigned else None)
                for row, item in enumerate(query)
            ),
        )
