from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import override

import numpy as np

from ...helpers import dist2sim, unbatchify_sim, unpack_float
from ...typing import AnySimFunc, ConversionFunc, Float, SimFunc
from .common import SequenceSim

__all__ = [
    "dtw",
    "twed",
    "smith_waterman",
]


def _drain[V](
    mapping: list[tuple[V | None, V | None]],
    similarities: list[float],
    x: Sequence[V] | np.ndarray,
    y: Sequence[V] | np.ndarray,
    i: int,
    j: int,
) -> tuple[list[tuple[V | None, V | None]], list[float]]:
    """Appends the unaligned remainder of either sequence and reverses the alignment.

    At most one of the two loops runs, and only once the other sequence is exhausted.
    """
    for index in range(i, 0, -1):
        mapping.append((None, x[index - 1]))
        similarities.append(0.0)

    for index in range(j, 0, -1):
        mapping.append((y[index - 1], None))
        similarities.append(0.0)

    return mapping[::-1], similarities[::-1]


class BaseElasticSimFunc[V](SimFunc[Sequence[V] | np.ndarray, SequenceSim[V, float]]):
    """Base class for the elastic measures, which align by minimizing a distance."""

    __slots__ = ()

    def compute(
        self,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
        return_alignment: bool,
    ) -> tuple[
        float,
        list[tuple[V | None, V | None]] | None,
        list[float] | None,
    ]:
        """Returns the distance and, if requested, the alignment it was taken from."""
        raise NotImplementedError

    def __call__(
        self,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
        return_alignment: bool = False,
    ) -> SequenceSim[V, float]:
        """
        Perform the alignment and optionally return alignment information.

        Args:
            x: The case sequence.
            y: The query sequence.
            return_alignment: Whether to compute and return the alignment.

        Returns:
            A SequenceSim object containing the similarity value, local similarities,
            and optional alignment.
        """
        distance, mapping, similarities = self.compute(x, y, return_alignment)

        return SequenceSim(
            value=dist2sim(distance),
            similarities=similarities,
            mapping=mapping,
        )


@dataclass(slots=True, init=False)
class dtw[V](BaseElasticSimFunc[V]):
    """
    Dynamic Time Warping similarity function with optional backtracking for alignment.

    Args:
        distance_func: Distance between two elements.
            Defaults to the absolute difference of numeric values.

    Examples:
        >>> sim = dtw()
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.5, similarities=None, mapping=None)
        >>> sim = dtw(distance_func=lambda a, b: abs(a - b))
        >>> sim([1, 2, 3], [3, 4, 5])
        SequenceSim(value=0.14285714285714285, similarities=None, mapping=None)
        >>> sim = dtw(distance_func=lambda a, b: abs(len(str(a)) - len(str(b))))
        >>> sim(["a", "bb", "ccc"], ["aa", "bbb", "c"], return_alignment=True)
        SequenceSim(value=0.25, similarities=[0.5, 1.0, 1.0, 0.3333333333333333],
            mapping=[('aa', 'a'), ('aa', 'bb'), ('bbb', 'ccc'), ('c', 'ccc')])
        >>> sim = dtw(distance_func=lambda a, b: abs(a - b))
        >>> sim([1, 2, 3], [1, 2, 3, 4], return_alignment=True)
        SequenceSim(value=0.5, similarities=[1.0, 1.0, 1.0, 0.5],
            mapping=[(1, 1), (2, 2), (3, 3), (4, 3)])
        >>> sim([], [1, 2], return_alignment=True)
        SequenceSim(value=0.0, similarities=[0.0, 0.0], mapping=[(1, None), (2, None)])
    """

    distance_func: SimFunc[V, Float] | None

    def __init__(self, distance_func: AnySimFunc[V, Float] | None = None):
        self.distance_func = unbatchify_sim(distance_func) if distance_func else None

    def distances(
        self, x: Sequence[V] | np.ndarray, y: Sequence[V] | np.ndarray
    ) -> np.ndarray:
        """Pairwise distances between all elements of the two sequences."""
        if self.distance_func is None:
            return np.abs(
                np.asarray(x, dtype=float)[:, None]
                - np.asarray(y, dtype=float)[None, :]
            )

        costs = np.empty((len(x), len(y)))

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                costs[i, j] = unpack_float(self.distance_func(xi, yj))

        return costs

    @override
    def compute(
        self,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
        return_alignment: bool,
    ) -> tuple[float, list[tuple[V | None, V | None]] | None, list[float] | None]:
        """Compute the DTW distance and optionally the alignment."""
        n, m = len(x), len(y)
        costs = self.distances(x, y)
        matrix = np.full((n + 1, m + 1), np.inf)
        matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matrix[i, j] = costs[i - 1, j - 1] + min(
                    matrix[i - 1, j - 1],  # Match
                    matrix[i - 1, j],  # Insertion
                    matrix[i, j - 1],  # Deletion
                )

        if not return_alignment:
            return float(matrix[n, m]), None, None

        mapping, similarities = self.backtrack(matrix, costs, x, y)

        return float(matrix[n, m]), mapping, similarities

    def backtrack(
        self,
        matrix: np.ndarray,
        costs: np.ndarray,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
    ) -> tuple[list[tuple[V | None, V | None]], list[float]]:
        """Backtrack through the cost matrix to obtain the alignment."""
        i, j = len(x), len(y)
        mapping: list[tuple[V | None, V | None]] = []
        similarities: list[float] = []

        while i > 0 and j > 0:
            # Align elements as (query, case)
            mapping.append((y[j - 1], x[i - 1]))
            similarities.append(dist2sim(float(costs[i - 1, j - 1])))

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


@dataclass(slots=True, init=False)
class twed[V](BaseElasticSimFunc[V]):
    """
    Time Warp Edit Distance (TWED) similarity function with optional backtracking.

    Unlike DTW, TWED is a true metric and satisfies the triangle inequality.
    Its edit operations are a *match* between two samples and a *delete* of a single
    sample, where every delete is charged the constant `penalty`.
    The elasticity along the time axis is controlled by `stiffness`: values close to
    zero behave like DTW, large values approach the Euclidean distance.

    Marteau, P.-F. (2009). Time Warp Edit Distance with Stiffness Adjustment for Time
    Series Matching. IEEE TPAMI, 31(2), 306-318. <https://arxiv.org/abs/cs/0703033>

    Args:
        distance_func: Distance between two samples.
            Defaults to the L2 norm of their difference, which is the absolute
            difference for scalar samples.
        timestamp_func: Timestamp of a sample, which must not decrease along the
            sequence. Defaults to the position of the sample within the sequence.
        stiffness: Elasticity along the time axis, called nu in the paper.
        penalty: Constant cost of a delete operation, called lambda in the paper.

    Examples:
        >>> sim = twed()
        >>> sim([1, 2, 3], [1, 2, 3])
        SequenceSim(value=1.0, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.33322225924691773, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [3, 4, 5])
        SequenceSim(value=0.11106175033318526, similarities=None, mapping=None)
        >>> sim([1, 2, 3], [1, 2, 3, 4], return_alignment=True)
        SequenceSim(value=0.33322225924691773, similarities=[1.0, 1.0, 1.0, 0.0],
            mapping=[(1, 1), (2, 2), (3, 3), (4, None)])
        >>> twed(stiffness=1.0)([1, 2, 3], [1, 2, 3, 4])
        SequenceSim(value=0.25, similarities=None, mapping=None)
        >>> sim = twed(distance_func=lambda a, b: abs(len(a) - len(b)))
        >>> sim(["a", "bb", "ccc"], ["aa", "bbb", "c"], return_alignment=True)
        SequenceSim(value=0.14285714285714285, similarities=[0.5, 0.5, 0.3333333333333333],
            mapping=[('aa', 'a'), ('bbb', 'bb'), ('c', 'ccc')])
        >>> sim = twed(
        ...     distance_func=lambda a, b: abs(a[1] - b[1]),
        ...     timestamp_func=lambda value: value[0],
        ... )
        >>> sim([(0, 1), (1, 2), (2, 3)], [(0, 1), (2, 3)], return_alignment=True)
        SequenceSim(value=0.24987506246876562, similarities=[1.0, 0.0, 1.0],
            mapping=[((0, 1), (0, 1)), (None, (1, 2)), ((2, 3), (2, 3))])
    """

    distance_func: SimFunc[V, Float] | None
    timestamp_func: ConversionFunc[V, float] | None
    stiffness: float
    penalty: float

    def __init__(
        self,
        distance_func: AnySimFunc[V, Float] | None = None,
        timestamp_func: ConversionFunc[V, float] | None = None,
        stiffness: float = 0.001,
        penalty: float = 1.0,
    ):
        if stiffness < 0.0:
            raise ValueError("The stiffness must not be negative")

        if penalty < 0.0:
            raise ValueError("The penalty must not be negative")

        self.distance_func = unbatchify_sim(distance_func) if distance_func else None
        self.timestamp_func = timestamp_func
        self.stiffness = stiffness
        self.penalty = penalty

    def timestamps(self, seq: Sequence[V] | np.ndarray) -> np.ndarray:
        """Timestamps of the sequence, prefixed by the virtual sample at time zero."""
        if self.timestamp_func is None:
            return np.concatenate([[0.0], np.arange(len(seq), dtype=float)])

        return np.array([0.0, *(float(self.timestamp_func(v)) for v in seq)])

    def distances(
        self, x: Sequence[V] | np.ndarray, y: Sequence[V] | np.ndarray
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
        n, m = len(x), len(y)
        pairwise_distances = np.zeros((n + 1, m + 1))
        consecutive_x = np.zeros(n + 1)
        consecutive_y = np.zeros(m + 1)

        if not n or not m:
            return pairwise_distances, consecutive_x, consecutive_y

        if self.distance_func is None:
            values_x = np.asarray(x, dtype=float).reshape(n, -1)
            values_y = np.asarray(y, dtype=float).reshape(m, -1)

            pairwise_distances[1:, 1:] = np.linalg.norm(
                values_x[:, None, :] - values_y[None, :, :], axis=-1
            )
            consecutive_x[2:] = np.linalg.norm(np.diff(values_x, axis=0), axis=-1)
            consecutive_y[2:] = np.linalg.norm(np.diff(values_y, axis=0), axis=-1)

            return pairwise_distances, consecutive_x, consecutive_y

        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                pairwise_distances[i + 1, j + 1] = unpack_float(
                    self.distance_func(xi, yj)
                )

        consecutive_x[2:] = [
            unpack_float(self.distance_func(a, b)) for a, b in pairwise(x)
        ]
        consecutive_y[2:] = [
            unpack_float(self.distance_func(a, b)) for a, b in pairwise(y)
        ]

        return pairwise_distances, consecutive_x, consecutive_y

    def costs(
        self, x: Sequence[V] | np.ndarray, y: Sequence[V] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cost of each edit operation for every cell of the matrix."""
        times_x, times_y = self.timestamps(x), self.timestamps(y)
        pairwise_distances, consecutive_x, consecutive_y = self.distances(x, y)

        deletion_x = np.zeros(len(x) + 1)
        deletion_x[1:] = (
            consecutive_x[1:] + self.stiffness * np.diff(times_x) + self.penalty
        )

        deletion_y = np.zeros(len(y) + 1)
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

        return pairwise_distances, match, deletion_x, deletion_y

    @override
    def compute(
        self,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
        return_alignment: bool,
    ) -> tuple[float, list[tuple[V | None, V | None]] | None, list[float] | None]:
        """Compute the TWED distance and optionally the alignment."""
        n, m = len(x), len(y)
        pairwise_distances, match, deletion_x, deletion_y = self.costs(x, y)

        matrix = np.full((n + 1, m + 1), np.inf)
        matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                matrix[i, j] = min(
                    matrix[i - 1, j - 1] + match[i - 1, j - 1],
                    matrix[i - 1, j] + deletion_x[i],
                    matrix[i, j - 1] + deletion_y[j],
                )

        if not return_alignment:
            return float(matrix[n, m]), None, None

        mapping, similarities = self.backtrack(
            matrix, match, deletion_x, deletion_y, pairwise_distances, x, y
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
    ) -> tuple[list[tuple[V | None, V | None]], list[float]]:
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
                similarities.append(dist2sim(float(pairwise_distances[i, j])))
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


@dataclass(slots=True, init=False)
class smith_waterman[V](SimFunc[Sequence[V], SequenceSim[V, float]]):
    """
    Smith-Waterman local alignment similarity function.

    Follows the ProCAKE formulation, where the scoring matrix is driven by a local
    similarity measure instead of fixed match and mismatch scores, the alignment is
    required to end with the last element of the query, and the raw score is
    normalized by the length of the query.

    Args:
        element_similarity: Similarity between two elements, expected to be in `[0, 1]`.
            Defaults to equality.
        deletion_penalty: Score of skipping an element of the query.
        insertion_penalty: Score of skipping an element of the case.

    Examples:
        >>> sim = smith_waterman()
        >>> sim("abcde", "abcde")
        SequenceSim(value=1.0, similarities=None, mapping=None)
        >>> sim("abcde", "fghe")
        SequenceSim(value=0.25, similarities=None, mapping=None)
        >>> sim("abcde", "xyz")
        SequenceSim(value=0.0, similarities=None, mapping=None)
        >>> sim("abcde", "abde", return_alignment=True)
        SequenceSim(value=0.75, similarities=[1.0, 1.0, 0.0, 1.0, 1.0],
            mapping=[('a', 'a'), ('b', 'b'), (None, 'c'), ('d', 'd'), ('e', 'e')])
    """

    element_similarity: SimFunc[V, Float] | None
    deletion_penalty: float
    insertion_penalty: float

    def __init__(
        self,
        element_similarity: AnySimFunc[V, Float] | None = None,
        deletion_penalty: float = -1.0,
        insertion_penalty: float = -1.0,
    ):
        self.element_similarity = (
            unbatchify_sim(element_similarity) if element_similarity else None
        )
        self.deletion_penalty = deletion_penalty
        self.insertion_penalty = insertion_penalty

    def __call__(
        self,
        x: Sequence[V],
        y: Sequence[V],
        return_alignment: bool = False,
    ) -> SequenceSim[V, float]:
        """
        Perform the alignment and optionally return alignment information.

        Args:
            x: The case sequence.
            y: The query sequence.
            return_alignment: Whether to compute and return the alignment.

        Returns:
            A SequenceSim object containing the similarity value, local similarities,
            and optional alignment.
        """
        if not x or not y:
            return SequenceSim(value=0.0)

        sims = self.similarities(x, y)
        matrix = self.compute_matrix(sims)
        value = min(1.0, max(0.0, float(matrix[len(y), :].max()) / len(y)))

        if not return_alignment:
            return SequenceSim(value=value)

        mapping, local_similarities = self.backtrack(matrix, sims, x, y)

        return SequenceSim(
            value=value,
            similarities=local_similarities,
            mapping=mapping,
        )

    def similarities(self, x: Sequence[V], y: Sequence[V]) -> np.ndarray:
        """Pairwise similarities with the query along the rows and the case along the columns."""
        sims = np.empty((len(y), len(x)))

        for i, yi in enumerate(y):
            for j, xj in enumerate(x):
                sims[i, j] = (
                    unpack_float(self.element_similarity(xj, yi))
                    if self.element_similarity
                    else float(xj == yi)
                )

        return sims

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
    ) -> tuple[list[tuple[V | None, V | None]], list[float]]:
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

        return mapping[::-1], similarities[::-1]
