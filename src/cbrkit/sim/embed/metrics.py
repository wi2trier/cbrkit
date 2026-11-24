"""Vector similarity metric functions (dense and sparse).

The dense metrics score whole columns of vectors with numpy (`cbrkit.typing.ColumnSimFunc`),
and `Cosine` and `Dot` also every case against every query at once (`cbrkit.typing.MatrixSimFunc`).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
import numpy.typing as npt

from ...schema import Family, serializable
from ...typing import ColumnSimFunc, MatrixSimFunc, NumpyArray, SimFunc, SparseVector


def _cosine(dots: NumpyArray, norms: NumpyArray) -> NumpyArray:
    """Cosine similarities from dot products and products of norms, 0 where a vector is zero."""
    return np.divide(dots, norms, out=np.zeros_like(norms), where=norms != 0.0)


def _scaled(values: NumpyArray, nonzero: npt.NDArray[np.bool_]) -> NumpyArray:
    """Values in [-1, 1] scaled to [0, 1], 0 where a vector is zero."""
    return np.where(nonzero, np.clip((values + 1.0) / 2.0, 0.0, 1.0), 0.0)


def _norms(cases: NumpyArray, queries: NumpyArray) -> NumpyArray:
    return np.linalg.norm(cases, axis=-1) * np.linalg.norm(queries, axis=-1)


def _stack(vectors: Sequence[NumpyArray]) -> NumpyArray:
    """The vectors as rows, a single row to broadcast if they are all the same, such as a query compared with many cases."""
    first = vectors[0]

    if all(vector is first for vector in vectors):
        return np.asarray(first)[None]

    return np.stack(vectors)


class _Dense(ColumnSimFunc[NumpyArray]):
    """Base of the metrics of dense vectors, which score whole columns of vectors with numpy."""

    __slots__ = ()

    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        """The similarities of the vectors along the last axis, broadcasting the others."""
        raise NotImplementedError

    @override
    def __call__(self, case: NumpyArray, query: NumpyArray) -> float:
        return float(self._rows(case, query))

    @override
    def columns(
        self, cases: Sequence[NumpyArray], queries: Sequence[NumpyArray]
    ) -> NumpyArray:
        if not cases:
            return np.zeros(0)

        return self._rows(np.stack(cases), _stack(queries))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Cosine(_Dense, MatrixSimFunc):
    """Cosine similarity for dense vectors, normalized to [0, 1].

    Examples:
        >>> Cosine()(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        0.5
        >>> Cosine().matrix(np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[1.0, 0.0]]))
        array([[1.],
               [0.]])
    """

    kind: Literal["cosine"] = field(default="cosine", repr=False)

    @override
    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        norms = _norms(cases, queries)

        return _scaled(_cosine((cases * queries).sum(axis=-1), norms), norms != 0.0)

    @override
    def matrix(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        norms = _norms(cases[:, None], queries)

        return _scaled(_cosine(cases @ queries.T, norms), norms != 0.0)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Dot(_Dense, MatrixSimFunc):
    """Dot product similarity for dense vectors, normalized to [0, 1]."""

    kind: Literal["dot"] = field(default="dot", repr=False)

    @override
    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        return _scaled(
            (cases * queries).sum(axis=-1),
            cases.any(axis=-1) & queries.any(axis=-1),
        )

    @override
    def matrix(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        return _scaled(
            cases @ queries.T, cases.any(axis=1, keepdims=True) & queries.any(axis=1)
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Angular(_Dense):
    """Angular similarity for dense vectors."""

    kind: Literal["angular"] = field(default="angular", repr=False)

    @override
    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        norms = _norms(cases, queries)
        cosine = _cosine((cases * queries).sum(axis=-1), norms)

        return np.where(
            norms != 0.0, 1.0 - np.arccos(np.clip(cosine, -1.0, 1.0)) / np.pi, 0.0
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Euclidean(_Dense):
    """Euclidean distance-based similarity for dense vectors."""

    kind: Literal["euclidean"] = field(default="euclidean", repr=False)

    @override
    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        return 1.0 / (1.0 + np.linalg.norm(cases - queries, axis=-1))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Manhattan(_Dense):
    """Manhattan distance-based similarity for dense vectors."""

    kind: Literal["manhattan"] = field(default="manhattan", repr=False)

    @override
    def _rows(self, cases: NumpyArray, queries: NumpyArray, /) -> NumpyArray:
        return 1.0 / (1.0 + np.abs(cases - queries).sum(axis=-1))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SparseDot(SimFunc[SparseVector, float]):
    """Dot product similarity for sparse vectors.

    Computes the dot product over shared dimensions and normalizes
    to [0, 1].  Returns 0.0 for empty vectors.

    Examples:
        >>> SparseDot()({0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0})
        1.0
        >>> SparseDot()({}, {0: 1.0})
        0.0
    """

    kind: Literal["sparse_dot"] = field(default="sparse_dot", repr=False)

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())
        normalized = (dot_prod + 1.0) / 2.0

        return max(0.0, min(1.0, normalized))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SparseCosine(SimFunc[SparseVector, float]):
    """Cosine similarity for sparse vectors.

    Computes cosine similarity between two sparse vectors and normalizes
    to [0, 1].  Returns 0.0 for empty vectors or zero norms.

    Examples:
        >>> SparseCosine()({0: 1.0, 1: 0.0}, {0: 1.0, 1: 0.0})
        1.0
        >>> SparseCosine()({0: 1.0}, {1: 1.0})
        0.5
        >>> SparseCosine()({}, {0: 1.0})
        0.0
    """

    kind: Literal["sparse_cosine"] = field(default="sparse_cosine", repr=False)

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())

        u_norm = math.hypot(*u.values())
        v_norm = math.hypot(*v.values())

        if u_norm == 0.0 or v_norm == 0.0:
            return 0.0

        cos_val = dot_prod / (u_norm * v_norm)
        cos_sim = (cos_val + 1.0) / 2.0

        return max(0.0, min(1.0, cos_sim))


__all__ = [
    "Angular",
    "Cosine",
    "Dot",
    "Euclidean",
    "Manhattan",
    "SparseCosine",
    "SparseDot",
]

VECTOR_METRICS = Family(
    "vector metric",
    (Cosine, Dot, Angular, Euclidean, Manhattan, SparseDot, SparseCosine),
)
