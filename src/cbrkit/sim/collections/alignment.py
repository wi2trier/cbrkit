from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Annotated, Literal, override

import numpy as np
from pydantic import Field

from ...helpers import batchify, dist2sim
from ...model.similarity import Alignment, SequenceSim
from ...schema import PythonOnly, serializable
from ...typing import Float, SimFunc
from .._nested import NestedMeasure
from ._common import empty_mapping, similarity_matrix

__all__ = [
    "DTW",
    "TWED",
    "SmithWaterman",
]


def _drain[V](
    mapping: list[tuple[V | None, V | None]],
    similarities: list[float],
    x: Sequence[V] | np.ndarray,
    y: Sequence[V] | np.ndarray,
    i: int,
    j: int,
) -> tuple[Alignment[V], tuple[float, ...]]:
    """Appends the unaligned remainder of either sequence and reverses the alignment.

    At most one of the two loops runs, and only once the other sequence is exhausted.
    """
    for index in range(i, 0, -1):
        mapping.append((None, x[index - 1]))
        similarities.append(0.0)

    for index in range(j, 0, -1):
        mapping.append((y[index - 1], None))
        similarities.append(0.0)

    return tuple(mapping[::-1]), tuple(similarities[::-1])


class BaseElasticSimFunc[V](SimFunc[Sequence[V] | np.ndarray, SequenceSim[V, float]]):
    """Base class for the elastic measures, which align by minimizing a distance."""

    __slots__ = ()

    def compute(
        self,
        case: Sequence[V] | np.ndarray,
        query: Sequence[V] | np.ndarray,
    ) -> tuple[float, Alignment[V] | None, tuple[float, ...] | None]:
        """Returns the distance and, if `alignment` is set, the alignment it was taken from."""
        raise NotImplementedError

    def to_similarity(
        self,
        distance: float,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
    ) -> float:
        """Converts the distance between the two sequences into a similarity."""
        return dist2sim(distance)

    def __call__(
        self,
        case: Sequence[V] | np.ndarray,
        query: Sequence[V] | np.ndarray,
    ) -> SequenceSim[V, float]:
        distance, mapping, similarities = self.compute(case, query)

        return SequenceSim(
            value=self.to_similarity(distance, case, query),
            similarities=similarities,
            mapping=mapping,
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class DTW[V](BaseElasticSimFunc[V]):
    """
    Dynamic Time Warping similarity function with optional backtracking for alignment.

    Args:
        element_measure: Similarity of two elements, whose distance is `1 - similarity`.
            Defaults to the absolute difference of numeric values as their distance.
        alignment: Whether the result carries the alignment and its local similarities,
            which are the similarities of the aligned elements.

    Examples:
        >>> sim = DTW()
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.5, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [3, 4, 5])
        SequenceSim(value=0.14285714285714285, similarities=None, mapping=None)
        >>> sim = DTW(alignment=True)
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.5, similarities=(1.0, 1.0, 1.0, 0.5),
            mapping=((1, 1), (2, 2), (3, 3), (4, 3)))
        >>> sim([], [1, 2])
        SequenceSim(value=0.0, similarities=(0.0, 0.0), mapping=((1, None), (2, None)))
        >>> from cbrkit.sim.generic import Equality
        >>> sim = DTW(element_measure=Equality(), alignment=True)
        >>> sim(["a", "b", "c"], ["a", "c"])
        SequenceSim(value=0.5, similarities=(1.0, 0.0, 1.0),
            mapping=(('a', 'a'), ('a', 'b'), ('c', 'c')))
    """

    element_measure: NestedMeasure[V, Float] | None = None
    alignment: bool = False
    kind: Literal["dtw"] = field(default="dtw", repr=False)

    def distances(
        self, case: Sequence[V] | np.ndarray, query: Sequence[V] | np.ndarray
    ) -> np.ndarray:
        """Pairwise distances between all elements of the two sequences."""
        if self.element_measure is None:
            return np.abs(
                np.asarray(case, dtype=float)[:, None]
                - np.asarray(query, dtype=float)[None, :]
            )

        return 1.0 - similarity_matrix(self.element_measure, case, query).floats.T

    def local_similarity(self, distance: float) -> float:
        """Similarity of two aligned elements given their distance."""
        return dist2sim(distance) if self.element_measure is None else 1.0 - distance

    @override
    def compute(
        self,
        case: Sequence[V] | np.ndarray,
        query: Sequence[V] | np.ndarray,
    ) -> tuple[float, Alignment[V] | None, tuple[float, ...] | None]:
        """Compute the DTW distance and optionally the alignment."""
        n, m = len(case), len(query)
        costs = self.distances(case, query)
        matrix = np.full((n + 1, m + 1), np.inf)
        matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matrix[i, j] = costs[i - 1, j - 1] + min(
                    matrix[i - 1, j - 1],  # Match
                    matrix[i - 1, j],  # Insertion
                    matrix[i, j - 1],  # Deletion
                )

        if not self.alignment:
            return float(matrix[n, m]), None, None

        mapping, similarities = self.backtrack(matrix, costs, case, query)

        return float(matrix[n, m]), mapping, similarities

    def backtrack(
        self,
        matrix: np.ndarray,
        costs: np.ndarray,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
    ) -> tuple[Alignment[V], tuple[float, ...]]:
        """Backtrack through the cost matrix to obtain the alignment."""
        i, j = len(x), len(y)
        mapping: list[tuple[V | None, V | None]] = []
        similarities: list[float] = []

        while i > 0 and j > 0:
            # Align elements as (query, case)
            mapping.append((y[j - 1], x[i - 1]))
            similarities.append(self.local_similarity(float(costs[i - 1, j - 1])))

            predecessors = (
                matrix[i - 1, j - 1],  # Match, preferred on ties
                matrix[i - 1, j],  # Insertion
                matrix[i, j - 1],  # Deletion
            )
            step = min(range(3), key=predecessors.__getitem__)

            if step == 0:
                i -= 1
                j -= 1

            elif step == 1:
                i -= 1

            else:
                j -= 1

        # The remainder is only non-empty when one of the sequences is
        return _drain(mapping, similarities, x, y, i, j)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class TWED[V](BaseElasticSimFunc[V]):
    """
    Time Warp Edit Distance (TWED) similarity function with optional backtracking.

    Unlike DTW, TWED is a true metric and satisfies the triangle inequality.
    Its edit operations are a `match` between two samples and a `delete` of a single
    sample, where every delete is charged the constant `penalty`.
    The elasticity along the time axis is controlled by `stiffness`: values close to
    zero behave like DTW, large values approach the Euclidean distance.

    By default, the distance is converted into a similarity via `1 / (1 + distance)`,
    which does not take the lengths of the sequences into account.
    If the sample distances lie in [0, 1], which is the case for an `element_measure`
    with similarities in [0, 1], `normalize=True` divides the distance by the largest
    value it can take for the two sequences instead, which is the distance obtained if
    all sample distances were 1, and returns `1 - distance / max_distance`.
    Identical sequences then have a similarity of 1, maximally different ones a
    similarity of 0, and the scale is comparable across sequence lengths.

    Marteau, P.-F. (2009). Time Warp Edit Distance with Stiffness Adjustment for Time
    Series Matching. IEEE TPAMI, 31(2), 306-318. <https://arxiv.org/abs/cs/0703033>

    Args:
        element_measure: Similarity of two samples, whose distance is `1 - similarity`.
            Defaults to the L2 norm of their difference as their distance, which is the
            absolute difference for scalar samples.
        timestamp: Timestamp of a sample, which must not decrease along the
            sequence. Defaults to the position of the sample within the sequence.
        stiffness: Elasticity along the time axis, called nu in the paper.
        penalty: Constant cost of a delete operation, called lambda in the paper.
        normalize: Normalize the distance by the largest distance the two sequences
            can have, which requires all sample distances to lie in [0, 1].
        alignment: Whether the result carries the alignment and its local similarities,
            which are `1 - distance` of the aligned samples with an `element_measure`
            or `normalize`, else `1 / (1 + distance)`.

    Examples:
        >>> sim = TWED()
        >>> sim([1, 2, 3], [1, 2, 3])
        SequenceSim(value=1.0, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.33322225924691773, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [3, 4, 5])
        SequenceSim(value=0.11106175033318526, similarities=None, mapping=None)
        >>> TWED(alignment=True)([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.33322225924691773, similarities=(1.0, 1.0, 1.0, 0.0),
            mapping=((1, 1), (2, 2), (3, 3), (4, None)))
        >>> TWED(stiffness=1.0)([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.25, similarities=None, mapping=None)
        >>> sim = TWED(
        ...     element_measure=lambda a, b: 1.0 - abs(a[1] - b[1]),
        ...     timestamp=lambda value: value[0],
        ...     alignment=True,
        ... )
        >>> sim([(0, 1), (1, 2), (2, 3)], [(0, 1), (2, 3)])
        SequenceSim(value=0.24987506246876562, similarities=(1.0, 0.0, 1.0),
            mapping=(((0, 1), (0, 1)), (None, (1, 2)), ((2, 3), (2, 3))))
        >>> from cbrkit.sim.generic import Equality
        >>> sim = TWED(element_measure=Equality(), normalize=True)
        >>> sim(["a", "b", "c", "d", "e"], ["a", "b", "x", "d", "e"])
        SequenceSim(value=0.7777777777777778, similarities=None, mapping=None)
    """

    element_measure: NestedMeasure[V, Float] | None = None
    timestamp: PythonOnly[Callable[[V], float] | None] = None
    stiffness: Annotated[float, Field(ge=0)] = 0.001
    penalty: Annotated[float, Field(ge=0)] = 1.0
    normalize: bool = False
    alignment: bool = False
    kind: Literal["twed"] = field(default="twed", repr=False)

    def timestamps(self, seq: Sequence[V] | np.ndarray) -> np.ndarray:
        """Timestamps of the sequence, prefixed by the virtual sample at time zero."""
        if self.timestamp is None:
            return np.concatenate([[0.0], np.arange(len(seq), dtype=float)])

        timestamps = np.array([0.0, *(float(self.timestamp(v)) for v in seq)])

        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) < 0.0):
            raise ValueError("Timestamps must be finite and nondecreasing")

        return timestamps

    def distances(
        self, case: Sequence[V] | np.ndarray, query: Sequence[V] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pairwise and consecutive sample distances, prefixed by the virtual sample.

        The paper prepends a sample with value zero at timestamp zero to both series.
        Since the first row and column of the cost matrix are infinite, the only cell
        reading a distance that involves this sample is the one where both virtual
        samples meet, which contributes zero.
        All entries at index zero are therefore zero, which also means that no zero
        element of the sample type is required.
        """
        n, m = len(case), len(query)
        pairwise_distances = np.zeros((n + 1, m + 1))
        consecutive_x = np.zeros(n + 1)
        consecutive_y = np.zeros(m + 1)

        if not n or not m:
            return pairwise_distances, consecutive_x, consecutive_y

        if self.element_measure is None:
            values_x = np.asarray(case, dtype=float).reshape(n, -1)
            values_y = np.asarray(query, dtype=float).reshape(m, -1)

            pairwise_distances[1:, 1:] = np.linalg.norm(
                values_x[:, None, :] - values_y[None, :, :], axis=-1
            )
            consecutive_x[2:] = np.linalg.norm(np.diff(values_x, axis=0), axis=-1)
            consecutive_y[2:] = np.linalg.norm(np.diff(values_y, axis=0), axis=-1)

            return pairwise_distances, consecutive_x, consecutive_y

        # All pairs in one batch: case by query, then the neighbors within each sequence.
        sims = batchify(self.element_measure)(
            [(x, y) for x in case for y in query] + [*pairwise(case), *pairwise(query)]
        )
        distances = 1.0 - np.array([float(sim) for sim in sims], dtype=float)
        pairwise_distances[1:, 1:] = distances[: n * m].reshape(n, m)
        consecutive_x[2:] = distances[n * m : n * m + n - 1]
        consecutive_y[2:] = distances[n * m + n - 1 :]

        return pairwise_distances, consecutive_x, consecutive_y

    def costs(
        self, case: Sequence[V] | np.ndarray, query: Sequence[V] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cost of each edit operation for every cell of the matrix."""
        times_x, times_y = self.timestamps(case), self.timestamps(query)
        pairwise_distances, consecutive_x, consecutive_y = self.distances(case, query)

        if self.normalize and any(
            np.any((distances < 0.0) | (distances > 1.0))
            for distances in (pairwise_distances, consecutive_x, consecutive_y)
        ):
            raise ValueError("With normalize=True, sample distances must lie in [0, 1]")

        return pairwise_distances, *self.edit_costs(
            times_x, times_y, pairwise_distances, consecutive_x, consecutive_y
        )

    def edit_costs(
        self,
        times_x: np.ndarray,
        times_y: np.ndarray,
        pairwise_distances: np.ndarray,
        consecutive_x: np.ndarray,
        consecutive_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cost of a match and of a delete in either sequence for every cell."""
        deletion_x = np.zeros(len(times_x))
        deletion_x[1:] = (
            consecutive_x[1:] + self.stiffness * np.diff(times_x) + self.penalty
        )

        deletion_y = np.zeros(len(times_y))
        deletion_y[1:] = (
            consecutive_y[1:] + self.stiffness * np.diff(times_y) + self.penalty
        )

        match = (
            pairwise_distances[1:, 1:]
            + pairwise_distances[:-1, :-1]
            + self.stiffness
            * (
                np.abs(times_x[1:, None] - times_y[None, 1:])
                + np.abs(times_x[:-1, None] - times_y[None, :-1])
            )
        )

        return match, deletion_x, deletion_y

    @staticmethod
    def cost_matrix(
        match: np.ndarray, deletion_x: np.ndarray, deletion_y: np.ndarray
    ) -> np.ndarray:
        """Accumulates the edit costs; the TWED distance is in the last cell."""
        n, m = len(deletion_x) - 1, len(deletion_y) - 1
        matrix = np.full((n + 1, m + 1), np.inf)
        matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matrix[i, j] = min(
                    matrix[i - 1, j - 1] + match[i - 1, j - 1],
                    matrix[i - 1, j] + deletion_x[i],
                    matrix[i, j - 1] + deletion_y[j],
                )

        return matrix

    def max_distance(
        self, case: Sequence[V] | np.ndarray, query: Sequence[V] | np.ndarray
    ) -> float:
        """
        Largest TWED distance the two sequences can have if all sample distances lie in [0, 1].

        The distance does not decrease when a sample distance increases, so it is largest
        if all sample distances are 1.
        With the default timestamps, this equals
        `2 * min(n, m) - 1 + abs(n - m) * (1 + stiffness + penalty)`.

        Examples:
            >>> TWED(stiffness=0.5).max_distance([1, 2, 3], [1, 2, 3, 4])
            7.5
        """
        n, m = len(case), len(query)
        pairwise_distances = np.zeros((n + 1, m + 1))
        pairwise_distances[1:, 1:] = 1.0
        consecutive_x = np.zeros(n + 1)
        consecutive_x[2:] = 1.0
        consecutive_y = np.zeros(m + 1)
        consecutive_y[2:] = 1.0

        match, deletion_x, deletion_y = self.edit_costs(
            self.timestamps(case),
            self.timestamps(query),
            pairwise_distances,
            consecutive_x,
            consecutive_y,
        )

        return float(self.cost_matrix(match, deletion_x, deletion_y)[n, m])

    @override
    def to_similarity(
        self,
        distance: float,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
    ) -> float:
        if not self.normalize:
            return dist2sim(distance)

        if not len(x) or not len(y):
            return 1.0 if len(x) == len(y) else 0.0

        return min(1.0, max(0.0, 1.0 - distance / self.max_distance(x, y)))

    def local_similarity(self, distance: float) -> float:
        """Similarity of two aligned samples given their distance."""
        if self.normalize or self.element_measure is not None:
            return 1.0 - distance

        return dist2sim(distance)

    @override
    def compute(
        self,
        case: Sequence[V] | np.ndarray,
        query: Sequence[V] | np.ndarray,
    ) -> tuple[float, Alignment[V] | None, tuple[float, ...] | None]:
        """Compute the TWED distance and optionally the alignment."""
        n, m = len(case), len(query)
        pairwise_distances, match, deletion_x, deletion_y = self.costs(case, query)
        matrix = self.cost_matrix(match, deletion_x, deletion_y)

        if not self.alignment:
            return float(matrix[n, m]), None, None

        mapping, similarities = self.backtrack(
            matrix, match, deletion_x, deletion_y, pairwise_distances, case, query
        )

        return float(matrix[n, m]), mapping, similarities

    def backtrack(
        self,
        matrix: np.ndarray,
        match: np.ndarray,
        deletion_x: np.ndarray,
        deletion_y: np.ndarray,
        pairwise_distances: np.ndarray,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
    ) -> tuple[Alignment[V], tuple[float, ...]]:
        """Backtrack through the cost matrix to obtain the alignment."""
        i, j = len(x), len(y)
        mapping: list[tuple[V | None, V | None]] = []
        similarities: list[float] = []

        while i > 0 and j > 0:
            candidates = (
                matrix[i - 1, j - 1] + match[i - 1, j - 1],  # Match, preferred on ties
                matrix[i - 1, j] + deletion_x[i],  # Delete in the case
                matrix[i, j - 1] + deletion_y[j],  # Delete in the query
            )
            step = min(range(3), key=candidates.__getitem__)

            if step == 0:
                mapping.append((y[j - 1], x[i - 1]))
                similarities.append(
                    self.local_similarity(float(pairwise_distances[i, j]))
                )
                i -= 1
                j -= 1

            elif step == 1:
                mapping.append((None, x[i - 1]))
                similarities.append(0.0)
                i -= 1

            else:
                mapping.append((y[j - 1], None))
                similarities.append(0.0)
                j -= 1

        return _drain(mapping, similarities, x, y, i, j)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SmithWaterman[V](SimFunc[Sequence[V], SequenceSim[V, float]]):
    """
    Smith-Waterman local alignment similarity function.

    Follows the ProCAKE formulation, where the scoring matrix is driven by a local
    similarity measure instead of fixed match and mismatch scores, the alignment is
    required to end with the last element of the query, and the raw score is
    normalized by the length of the query.

    Args:
        element_measure: Similarity between two elements, expected to be in `[0, 1]`.
            Defaults to equality.
        deletion_penalty: Score of skipping an element of the query.
        insertion_penalty: Score of skipping an element of the case.
        alignment: Whether the result carries the alignment and its local similarities.

    Examples:
        >>> sim = SmithWaterman()
        >>> sim("abcde", "abcde")
        SequenceSim(value=1.0, similarities=None, mapping=None)
        >>> sim("abcde", "fghe")
        SequenceSim(value=0.25, similarities=None, mapping=None)
        >>> sim("abcde", "xyz")
        SequenceSim(value=0.0, similarities=None, mapping=None)
        >>> SmithWaterman(alignment=True)("abcde", "abde")
        SequenceSim(value=0.75, similarities=(1.0, 1.0, 0.0, 1.0, 1.0),
            mapping=(('a', 'a'), ('b', 'b'), (None, 'c'), ('d', 'd'), ('e', 'e')))
    """

    element_measure: NestedMeasure[V, Float] | None = None
    deletion_penalty: float = -1.0
    insertion_penalty: float = -1.0
    alignment: bool = False
    kind: Literal["smith_waterman"] = field(default="smith_waterman", repr=False)

    @override
    def __call__(self, case: Sequence[V], query: Sequence[V]) -> SequenceSim[V, float]:
        if (empty := empty_mapping(case, query)) is not None:
            return empty

        sims = self.similarities(case, query)
        matrix = self.compute_matrix(sims)
        value = min(1.0, max(0.0, float(matrix[len(query), :].max()) / len(query)))

        if not self.alignment:
            return SequenceSim(value=value)

        mapping, local_similarities = self.backtrack(matrix, sims, case, query)

        return SequenceSim(
            value=value,
            similarities=local_similarities,
            mapping=mapping,
        )

    def similarities(self, case: Sequence[V], query: Sequence[V]) -> np.ndarray:
        """Pairwise similarities with the query along the rows and the case along the columns."""
        if self.element_measure is None:
            return np.array([[float(x == y) for x in case] for y in query], dtype=float)

        return similarity_matrix(self.element_measure, case, query).floats

    def compute_matrix(self, sims: np.ndarray) -> np.ndarray:
        """Fill the scoring matrix, where negative scores are clamped to zero."""
        m, n = sims.shape
        matrix = np.zeros((m + 1, n + 1))

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                matrix[i, j] = max(
                    0.0,
                    matrix[i - 1, j - 1] + sims[i - 1, j - 1],
                    matrix[i - 1, j] + self.deletion_penalty,
                    matrix[i, j - 1] + self.insertion_penalty,
                )

        return matrix

    def backtrack(
        self,
        matrix: np.ndarray,
        sims: np.ndarray,
        x: Sequence[V],
        y: Sequence[V],
    ) -> tuple[Alignment[V], tuple[float, ...]]:
        """Backtrack from the best cell of the last query row to the first zero cell."""
        i = len(y)
        j = int(np.argmax(matrix[i, :]))
        mapping: list[tuple[V | None, V | None]] = []
        similarities: list[float] = []

        while i > 0 and j > 0 and matrix[i, j] > 0.0:
            candidates = (
                matrix[i - 1, j - 1] + sims[i - 1, j - 1],  # Match, preferred on ties
                matrix[i - 1, j] + self.deletion_penalty,
                matrix[i, j - 1] + self.insertion_penalty,
            )
            step = max(range(3), key=candidates.__getitem__)

            if step == 0:
                mapping.append((y[i - 1], x[j - 1]))
                similarities.append(float(sims[i - 1, j - 1]))
                i -= 1
                j -= 1

            elif step == 1:
                mapping.append((y[i - 1], None))
                similarities.append(0.0)
                i -= 1

            else:
                mapping.append((None, x[j - 1]))
                similarities.append(0.0)
                j -= 1

        return tuple(mapping[::-1]), tuple(similarities[::-1])
