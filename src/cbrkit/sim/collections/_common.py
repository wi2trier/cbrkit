from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from ...helpers import batchify
from ...model.similarity import SequenceSim
from ...typing import AnySimFunc, Float, NumpyArray

__all__ = ["Matrix", "empty_mapping", "similarity_matrix"]


@dataclass(slots=True, frozen=True)
class Matrix[S: Float]:
    """Similarities of all element pairs with a row per query element and a column per case element.

    Args:
        values: The similarities row by row as returned by the measure.
        floats: The similarities as floats.
    """

    values: Sequence[S]
    floats: NumpyArray

    def __getitem__(self, index: tuple[int, int]) -> S:
        row, col = index

        return self.values[row * self.floats.shape[1] + col]


def similarity_matrix[V, S: Float](
    measure: AnySimFunc[V, S], case: Iterable[V], query: Iterable[V]
) -> Matrix[S]:
    """Similarities of all element pairs, computed in one batch with the case element first."""
    cases, queries = list(case), list(query)
    values = batchify(measure)([(x, y) for y in queries for x in cases])

    return Matrix(
        values,
        np.array([float(value) for value in values], dtype=float).reshape(
            len(queries), len(cases)
        ),
    )


def empty_mapping[V, S: Float](
    case: Sequence[V], query: Sequence[V]
) -> SequenceSim[V, S] | None:
    """The result for an empty sequence, None if neither is empty.

    Two empty sequences are identical, and an empty one has nothing in common with any other.
    """
    if case and query:
        return None

    return SequenceSim(
        value=float(len(case) == len(query)), similarities=(), mapping=()
    )
