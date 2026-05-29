"""Shared helpers and mixins for indexable retrievers."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, override

from ...filter import Filter
from ...helpers import batchify_sim, dispatch_batches, dispatch_batches_async
from ...sim.embed import default_score_func, embed_pairs
from ...typing import (
    AsyncRetrieverFunc,
    BatchConversionFunc,
    Casebase,
    NumpyArray,
    RetrieverFunc,
    SimMap,
)

type SearchType = Literal["dense", "sparse", "hybrid"]

_EMPTY_CASEBASE_MSG = (
    "Indexed retrieval was requested with an empty casebase, "
    "but no index is available. Call put_index() first."
)
_NO_BRUTE_MSG = (
    "Brute-force search is not supported for search_type={search_type!r}. "
    "Call put_index() first."
)


def resolve_casebases[K, V](
    batches: Sequence[tuple[Casebase[K, V], V]],
    indexed_casebase: Casebase[K, V] | None,
) -> list[tuple[Casebase[K, V], V]]:
    """Resolve casebases for indexable retrievers.

    Empty casebases are treated as an explicit signal to use indexed retrieval mode.
    In indexed mode, empty casebases are replaced with the previously indexed casebase.
    """
    if indexed_casebase is None:
        if any(len(casebase) == 0 for casebase, _ in batches):
            raise ValueError(
                "Indexed retrieval was requested with an empty casebase, but no index is available. "
                "Call put_index() first."
            )

        return list(batches)

    return [
        (indexed_casebase if len(casebase) == 0 else casebase, query)
        for casebase, query in batches
    ]


def _normalize_results[K](
    results: Sequence[tuple[Casebase[K, str], SimMap[K, float]]],
    enabled: bool,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Apply per-query min-max normalization if enabled.

    When all scores in a query coincide (e.g. single-hit queries), every
    entry is mapped to 1.0 — the best score — rather than 0.0, which
    would silently mark a perfect single match as a non-match.
    """
    if not enabled:
        return results

    normalized: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for cb, sm in results:
        if not sm:
            normalized.append((cb, sm))
            continue

        mn = min(sm.values())
        mx = max(sm.values())

        if mn == mx:
            normalized.append((cb, {k: 1.0 for k in sm}))
            continue

        spread = mx - mn
        normalized.append((cb, {k: (v - mn) / spread for k, v in sm.items()}))

    return normalized


def _brute_force_dense_search[K](
    queries: Sequence[str],
    casebase: Casebase[K, str],
    conversion_func: BatchConversionFunc[str, NumpyArray],
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Shared brute-force dense vector search for non-indexed casebases."""
    keys = list(casebase.keys())

    case_vecs, query_vecs = embed_pairs(
        conversion_func, query_conversion_func, list(casebase.values()), list(queries)
    )

    sim_func = batchify_sim(default_score_func)

    results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for qvec in query_vecs:
        sims = sim_func([(cv, qvec) for cv in case_vecs])
        results.append(
            (
                dict(casebase),
                dict(zip(keys, sims, strict=True)),
            )
        )

    return results


@dataclass(slots=True, kw_only=True)
class RrfMixin:
    """Reciprocal Rank Fusion parameters shared across hybrid retrievers.

    Keyword-only so subclass dataclasses can keep positional fields
    without conflicting with the defaults declared here.

    This is the slotted *root* of both storage-retriever base trees
    (``StorageRetriever`` and ``AsyncVectorStorageRetriever`` extend it),
    so it stays in a single linear MRO rather than being a slotted sibling
    — two slotted sibling bases cannot share a C-level instance layout.
    The parameters only affect backends that fuse rankings via RRF
    (``rrf_k`` for zvec; ``rrf_k`` + ``rrf_weights`` for chromadb and
    pgvector) and are inert for backends with a native hybrid reranker.
    """

    rrf_k: int = 60
    """Smoothing parameter in the RRF denominator: ``1 / (rrf_k + rank)``."""

    rrf_weights: tuple[float, float] = (0.7, 0.3)
    """Weights for ``(dense, sparse)`` rankings in the fusion sum."""


def reciprocal_rank_fusion[K, V](
    rankings: Sequence[Iterable[tuple[K, V]]],
    weights: Sequence[float],
    rrf_k: int,
) -> tuple[dict[K, float], dict[K, V]]:
    """Combine multiple ranked lists into RRF-fused scores.

    Each ranking is an iterable of ``(key, value)`` pairs in ranked
    order (rank 1 first).  Returns ``(scores, values)`` where
    ``scores[key] = Σ wᵢ / (rrf_k + rankᵢ)`` and ``values[key]`` is the
    first value seen for that key across rankings.
    """
    assert len(rankings) == len(weights)
    scores: dict[K, float] = {}
    values: dict[K, V] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (key, value) in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            values.setdefault(key, value)
    return scores, values


class _IndexedStorage(Protocol):
    """Minimal storage surface the dispatch skeleton relies on."""

    def has_index(self) -> bool: ...


class _VectorStorage(_IndexedStorage, Protocol):
    """Indexed storage that can also embed text for brute-force search."""

    conversion_func: BatchConversionFunc[str, NumpyArray] | None

    def search_limit(self) -> int | None: ...


class _AsyncIndexedStorage(Protocol):
    async def has_index(self) -> bool: ...


class _AsyncVectorStorage(_AsyncIndexedStorage, Protocol):
    conversion_func: BatchConversionFunc[str, NumpyArray] | None
    index_type: SearchType


class _AsyncSqlVectorStorage(_AsyncVectorStorage, Protocol):
    """Async vector storage whose DB keys are cast back to casebase keys."""

    def cast_key(self, value: Any) -> Any: ...


@dataclass(slots=True)
class StorageRetriever[K, St: _IndexedStorage](
    RrfMixin, RetrieverFunc[K, str, float], ABC
):
    """Sync dispatch skeleton for storage-backed retrievers.

    Owns the parts every storage retriever shares: empty-casebase routing
    to the index, the brute-force fallback for non-indexed casebases, and
    shared-casebase batching via :func:`dispatch_batches`.

    Subclasses must implement:

    * :meth:`_search_db` — query the persisted index for *queries*.
    * :meth:`_search_brute` — rank an in-memory *casebase* with no index.

    Most vector backends should extend :class:`VectorStorageRetriever`
    instead, which implements both in terms of three search-type leaves.
    """

    storage: St
    limit: int | None = None
    normalize_scores: bool = True

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if not batches:
            return []
        return dispatch_batches(batches, self._dispatch)

    def _dispatch(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if len(casebase) == 0:
            if not self.storage.has_index():
                raise ValueError(_EMPTY_CASEBASE_MSG)
            return self._search_db(queries)
        return self._search_brute(queries, casebase)

    @abstractmethod
    def _search_db(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Query the persisted index for *queries*."""
        ...

    @abstractmethod
    def _search_brute(
        self, queries: Sequence[str], casebase: Casebase[K, str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Rank an in-memory *casebase* with no index available."""
        ...


@dataclass(slots=True)
class VectorStorageRetriever[K, St: _VectorStorage](StorageRetriever[K, St], ABC):
    """Dispatch skeleton for vector backends with dense/sparse/hybrid search.

    Implements :meth:`_search_db` (search-type dispatch + normalization) and
    :meth:`_search_brute` (dense brute-force over an in-memory casebase) in
    terms of three leaves subclasses provide:

    * :meth:`_search_db_dense`, :meth:`_search_db_sparse`,
      :meth:`_search_db_hybrid`.

    Override :meth:`_effective_search_type` when the active search type is
    resolved dynamically rather than taken verbatim from *search_type*.
    """

    search_type: SearchType = "dense"
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None

    def _clamp_search_limit(self) -> int | None:
        """Resolve the effective query limit, clamped to the storage cap.

        Returns the smaller of the configured ``limit`` and the storage's
        ``search_limit()``, or ``None`` when neither bounds the query.
        Backends that require a finite ``k`` (e.g. ``zvec``) apply their own
        default to the result.
        """
        n = self.storage.search_limit()
        if n is None:
            return self.limit
        if self.limit is not None:
            return min(self.limit, n)
        return n

    def _effective_search_type(self) -> SearchType:
        """The search type to run (verbatim by default)."""
        return self.search_type

    def _search_db(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        match self._effective_search_type():
            case "dense":
                results = self._search_db_dense(queries)
            case "sparse":
                results = self._search_db_sparse(queries)
            case "hybrid":
                results = self._search_db_hybrid(queries)
        return _normalize_results(results, self.normalize_scores)

    def _search_brute(
        self, queries: Sequence[str], casebase: Casebase[K, str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        search_type = self._effective_search_type()
        if search_type != "dense":
            raise NotImplementedError(_NO_BRUTE_MSG.format(search_type=search_type))
        assert self.storage.conversion_func is not None
        return _brute_force_dense_search(
            queries,
            casebase,
            self.storage.conversion_func,
            self.query_conversion_func,
        )

    @abstractmethod
    def _search_db_dense(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...

    @abstractmethod
    def _search_db_sparse(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...

    @abstractmethod
    def _search_db_hybrid(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...


@dataclass(slots=True)
class AsyncVectorStorageRetriever[K, St: _AsyncVectorStorage](
    RrfMixin, AsyncRetrieverFunc[K, str, float], ABC
):
    """Async mirror of :class:`VectorStorageRetriever`.

    Same skeleton and leaves, but ``async`` throughout and batched via
    :func:`dispatch_batches_async`.  *search_type* may be ``None`` to defer
    to the storage's ``index_type`` (see :meth:`_effective_search_type`).
    """

    storage: St
    search_type: SearchType | None = None
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    normalize_scores: bool = True

    async def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if not batches:
            return []
        return list(await dispatch_batches_async(batches, self._dispatch))

    def _effective_search_type(self) -> SearchType:
        """Explicit *search_type*, else the storage's ``index_type``."""
        return self.search_type if self.search_type is not None else self.storage.index_type

    async def _dispatch(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if len(casebase) == 0:
            if not await self.storage.has_index():
                raise ValueError(_EMPTY_CASEBASE_MSG)
            return await self._search_db(queries)
        return await self._search_brute(queries, casebase)

    async def _search_db(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        match self._effective_search_type():
            case "dense":
                results = await self._search_db_dense(queries)
            case "sparse":
                results = await self._search_db_sparse(queries)
            case "hybrid":
                results = await self._search_db_hybrid(queries)
        return _normalize_results(results, self.normalize_scores)

    async def _search_brute(
        self, queries: Sequence[str], casebase: Casebase[K, str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        search_type = self._effective_search_type()
        if search_type != "dense":
            raise NotImplementedError(_NO_BRUTE_MSG.format(search_type=search_type))
        assert self.storage.conversion_func is not None
        results = await asyncio.to_thread(
            _brute_force_dense_search,
            queries,
            casebase,
            self.storage.conversion_func,
            self.query_conversion_func,
        )
        return cast("Sequence[tuple[Casebase[K, str], SimMap[K, float]]]", results)

    @abstractmethod
    async def _search_db_dense(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...

    @abstractmethod
    async def _search_db_sparse(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...

    @abstractmethod
    async def _search_db_hybrid(
        self, queries: Sequence[str]
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]: ...


@dataclass(slots=True)
class SqlAlchemyVectorRetriever[K, St: _AsyncSqlVectorStorage](
    AsyncVectorStorageRetriever[K, St], ABC
):
    """Async vector retriever over a SQLAlchemy-backed storage.

    Adds the filter and hybrid-oversample knobs plus the row-collection and
    RRF-finalization helpers that the ``pgvector`` and ``sqlite_vec`` backends
    use identically — only the three ``_search_db_*`` leaves differ between
    them.
    """

    where: Filter | None = None
    hybrid_oversample: int = 4

    async def _collect_rows(
        self,
        conn: Any,
        stmt: Any,
        score_fn: Callable[[float], float],
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        """Run *stmt* and fold its ``(key, value, score)`` rows into a result pair."""
        cast_key = self.storage.cast_key
        cb: dict[K, str] = {}
        sm: dict[K, float] = {}
        for k, v, s in (await conn.execute(stmt)).all():
            kk = cast_key(k)
            cb[kk] = v
            sm[kk] = score_fn(float(s))
        return cb, sm

    def _finalize_rrf(
        self,
        scores: Mapping[K, float],
        values: Mapping[K, str],
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        """Sort RRF *scores* descending, trim to ``limit``, project *values*."""
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        if self.limit is not None:
            ranked = ranked[: self.limit]
        return {k: values[k] for k, _ in ranked}, dict(ranked)


__all__ = [
    "resolve_casebases",
    "_normalize_results",
    "_brute_force_dense_search",
    "RrfMixin",
    "reciprocal_rank_fusion",
    "SearchType",
    "StorageRetriever",
    "VectorStorageRetriever",
    "AsyncVectorStorageRetriever",
    "SqlAlchemyVectorRetriever",
]
