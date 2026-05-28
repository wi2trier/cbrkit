"""Vector similarity metric functions (dense and sparse)."""

from dataclasses import dataclass
from typing import override

import numpy as np

from ...typing import NumpyArray, SimFunc, SparseVector


@dataclass(slots=True, frozen=True)
class cosine(SimFunc[NumpyArray, float]):
    """Cosine similarity for dense vectors, normalized to [0, 1]."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)

            if u_norm == 0.0 or v_norm == 0.0:
                return 0.0

            # [-1, 1]
            cos_val = np.dot(u, v) / (u_norm * v_norm)

            # [0, 1]
            cos_sim = (cos_val + 1.0) / 2.0

            return np.clip(cos_sim, 0.0, 1.0).__float__()

        return 0.0


@dataclass(slots=True, frozen=True)
class dot(SimFunc[NumpyArray, float]):
    """Dot product similarity for dense vectors, normalized to [0, 1]."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            dot_prod = (np.dot(u, v) + 1.0) / 2.0

            return np.clip(dot_prod, 0.0, 1.0).__float__()

        return 0.0


@dataclass(slots=True, frozen=True)
class angular(SimFunc[NumpyArray, float]):
    """Angular similarity for dense vectors."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            try:
                return (
                    1.0
                    - np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                    / np.pi
                )
            except Exception:
                pass

        return 0.0


@dataclass(slots=True, frozen=True)
class euclidean(SimFunc[NumpyArray, float]):
    """Euclidean distance-based similarity for dense vectors."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.linalg.norm(u - v).__float__())


@dataclass(slots=True, frozen=True)
class manhattan(SimFunc[NumpyArray, float]):
    """Manhattan distance-based similarity for dense vectors."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.sum(np.abs(u - v)).__float__())


@dataclass(slots=True, frozen=True)
class sparse_dot(SimFunc[SparseVector, float]):
    """Dot product similarity for sparse vectors.

    Computes the dot product over shared dimensions and normalizes
    to [0, 1].  Returns 0.0 for empty vectors.

    Examples:
        >>> sparse_dot()({0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0})
        1.0
        >>> sparse_dot()({}, {0: 1.0})
        0.0
    """

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())
        normalized = (dot_prod + 1.0) / 2.0

        return max(0.0, min(1.0, normalized))


@dataclass(slots=True, frozen=True)
class sparse_cosine(SimFunc[SparseVector, float]):
    """Cosine similarity for sparse vectors.

    Computes cosine similarity between two sparse vectors and normalizes
    to [0, 1].  Returns 0.0 for empty vectors or zero norms.

    Examples:
        >>> sparse_cosine()({0: 1.0, 1: 0.0}, {0: 1.0, 1: 0.0})
        1.0
        >>> sparse_cosine()({0: 1.0}, {1: 1.0})
        0.5
        >>> sparse_cosine()({}, {0: 1.0})
        0.0
    """

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())

        u_norm = sum(val * val for val in u.values()) ** 0.5
        v_norm = sum(val * val for val in v.values()) ** 0.5

        if u_norm == 0.0 or v_norm == 0.0:
            return 0.0

        cos_val = dot_prod / (u_norm * v_norm)
        cos_sim = (cos_val + 1.0) / 2.0

        return max(0.0, min(1.0, cos_sim))


default_score_func: SimFunc[NumpyArray, float] = cosine()


__all__ = [
    "cosine",
    "dot",
    "angular",
    "euclidean",
    "manhattan",
    "sparse_dot",
    "sparse_cosine",
    "default_score_func",
]
