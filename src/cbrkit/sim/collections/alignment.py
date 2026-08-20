from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...helpers import dist2sim, unbatchify_sim, unpack_float
from ...typing import AnySimFunc, Float, SimFunc
from .common import SequenceSim

__all__ = [
    "dtw",
    "smith_waterman",
]


@dataclass(slots=True, init=False)
class dtw[V](SimFunc[Sequence[V] | np.ndarray, SequenceSim[V, float]]):
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

    def __call__(
        self,
        x: Sequence[V] | np.ndarray,
        y: Sequence[V] | np.ndarray,
        return_alignment: bool = False,
    ) -> SequenceSim[V, float]:
        """
        Perform DTW and optionally return alignment information.

        Args:
            x: The case sequence.
            y: The query sequence.
            return_alignment: Whether to compute and return the alignment.

        Returns:
            A SequenceSim object containing the similarity value, local similarities,
            and optional alignment.
        """
        distance, mapping, similarities = self.compute_dtw(x, y, return_alignment)

        return SequenceSim(
            value=dist2sim(distance),
            similarities=similarities,
            mapping=mapping,
        )

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

    def compute_dtw(
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

        # Only reachable when one of the sequences is empty
        while i > 0:
            mapping.append((None, x[i - 1]))
            similarities.append(0.0)
            i -= 1

        while j > 0:
            mapping.append((y[j - 1], None))
            similarities.append(0.0)
            j -= 1

        return mapping[::-1], similarities[::-1]


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
