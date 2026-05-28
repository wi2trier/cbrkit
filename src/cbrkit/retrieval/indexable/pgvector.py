"""PostgreSQL/pgvector retriever wrappers (sync + async).

These retrievers query strings against a designated text column on a
:class:`cbrkit.indexable.pgvector_async` storage.  They satisfy
:class:`cbrkit.typing.AsyncRetrieverFunc` / ``RetrieverFunc`` over
``str`` queries and return ``Casebase[K, str]`` — the text column
projection.  Users wanting the full tabular row read it from the
storage in a follow-up call.

The retriever is a pure query path: index maintenance lives on the
storage that owns the index (call ``storage.put_index(...)``,
``storage.upsert_index(...)`` etc., or pass the storage to
:func:`cbrkit.retain.indexable`).
"""

import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from ...filter import Filter
from ...helpers import run_coroutine
from ...indexable import (
    pgvector as pgvector_storage,
    pgvector_async as pgvector_async_storage,
)
from ...indexable._common import PG_METRICS
from ...typing import (
    AsyncRetrieverFunc,
    BatchConversionFunc,
    Casebase,
    NumpyArray,
    RetrieverFunc,
    SimMap,
)
from ._common import (
    RrfMixin,
    _brute_force_dense_search,
    _normalize_results,
    reciprocal_rank_fusion,
)


def _build_dense_stmt(
    storage: pgvector_async_storage[Any, Any],
    qvec: list[float],
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    assert storage.text_column is not None
    table = storage.sa_table
    metric = PG_METRICS[storage.metric_type]
    distance = getattr(table.c[storage.pgvector_column], metric.distance_attr)(qvec)
    stmt = sa.select(
        table.c[storage.key_column],
        table.c[storage.text_column],
        distance.label("distance"),
    ).order_by(distance)
    if where is not None:
        stmt = stmt.where(storage.compile_filter(where))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def _build_sparse_stmt(
    storage: pgvector_async_storage[Any, Any],
    query: str,
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    assert storage.text_column is not None
    table = storage.sa_table
    tsv = table.c[storage.tsvector_column]
    tsq = sa.func.plainto_tsquery(storage.tsvector_config, query)
    rank = sa.func.ts_rank(tsv, tsq)
    stmt = (
        sa.select(
            table.c[storage.key_column],
            table.c[storage.text_column],
            rank.label("score"),
        )
        .where(tsv.op("@@")(tsq))
        .order_by(rank.desc())
    )
    if where is not None:
        stmt = stmt.where(storage.compile_filter(where))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


@dataclass(slots=True)
class pgvector_async[K: int | str](
    AsyncRetrieverFunc[K, str, float],
    RrfMixin,
):
    """Async retriever wrapper for :class:`cbrkit.indexable.pgvector_async`.

    Queries are strings, matched against the storage's ``text_column``
    via dense (HNSW), sparse (GIN/FTS), or hybrid (RRF) search.  The
    returned casebase contains the text-column values; for full row
    data, read from the storage separately.

    *search_type* defaults to ``None`` and resolves to the storage's
    ``index_type`` at call time — set it explicitly only to query a
    subset of what was indexed (e.g. ``"dense"`` on a hybrid index).
    """

    storage: pgvector_async_storage[K, Any]
    search_type: Literal["dense", "sparse", "hybrid"] | None = None
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: Filter | None = None
    hybrid_oversample: int = 4
    normalize_scores: bool = True

    @property
    def _resolved_search_type(self) -> Literal["dense", "sparse", "hybrid"]:
        """Effective search type — explicit override or storage's index_type."""
        return self.search_type if self.search_type is not None else self.storage.index_type

    async def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if not batches:
            return []

        async def per_batch(
            casebase: Casebase[K, str], query: str
        ) -> tuple[Casebase[K, str], SimMap[K, float]]:
            if len(casebase) == 0:
                if not await self.storage.has_index():
                    raise ValueError(
                        "Indexed retrieval was requested with an empty casebase, "
                        "but no index is available. Call storage.put_index() first."
                    )
                return (await self._search_db([query]))[0]
            return (await self._search_brute([query], casebase))[0]

        results = await asyncio.gather(*(per_batch(cb, q) for cb, q in batches))
        return list(results)

    # -- internal search helpers --------------------------------------------

    async def _search_brute(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        search_type = self._resolved_search_type
        if search_type != "dense":
            raise NotImplementedError(
                f"Brute-force search is not supported for search_type="
                f"{search_type!r}. Call storage.put_index() first."
            )
        assert self.storage.conversion_func is not None
        results = await asyncio.to_thread(
            _brute_force_dense_search,
            queries,
            casebase,
            self.storage.conversion_func,
            self.query_conversion_func,
        )
        return cast("Sequence[tuple[Casebase[K, str], SimMap[K, float]]]", results)

    async def _search_db(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        match self._resolved_search_type:
            case "dense":
                results = await self._search_db_dense(queries)
            case "sparse":
                results = await self._search_db_sparse(queries)
            case "hybrid":
                results = await self._search_db_hybrid(queries)
        return _normalize_results(results, self.normalize_scores)

    async def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage.conversion_func is not None
        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = await asyncio.to_thread(embed_func, list(queries))
        score_fn = PG_METRICS[self.storage.metric_type].sim_from_distance
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.sa_engine.connect() as conn:
            for qvec in query_vecs:
                stmt = _build_dense_stmt(
                    self.storage, np.asarray(qvec).tolist(), self.where, self.limit
                )
                results.append(await self._collect_rows(conn, stmt, score_fn))

        return results

    async def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.sa_engine.connect() as conn:
            for query in queries:
                stmt = _build_sparse_stmt(self.storage, query, self.where, self.limit)
                results.append(await self._collect_rows(conn, stmt, lambda s: s))

        return results

    async def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage.conversion_func is not None
        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = await asyncio.to_thread(embed_func, list(queries))
        candidate_n = (
            self.limit * self.hybrid_oversample if self.limit is not None else None
        )
        cast_key = self.storage.cast_key
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.sa_engine.connect() as conn:
            for query, qvec in zip(queries, query_vecs, strict=True):
                dense_rows = (
                    await conn.execute(
                        _build_dense_stmt(
                            self.storage,
                            np.asarray(qvec).tolist(),
                            self.where,
                            candidate_n,
                        )
                    )
                ).all()
                sparse_rows = (
                    await conn.execute(
                        _build_sparse_stmt(
                            self.storage, query, self.where, candidate_n
                        )
                    )
                ).all()

                scores, values = reciprocal_rank_fusion(
                    [
                        ((cast_key(k), v) for k, v, _ in dense_rows),
                        ((cast_key(k), v) for k, v, _ in sparse_rows),
                    ],
                    self.rrf_weights,
                    self.rrf_k,
                )

                ranked = sorted(scores.items(), key=lambda kv: -kv[1])
                if self.limit is not None:
                    ranked = ranked[: self.limit]

                results.append(
                    ({k: values[k] for k, _ in ranked}, dict(ranked))
                )

        return results

    async def _collect_rows(
        self,
        conn: AsyncConnection,
        stmt: sa.Select[Any],
        score_fn: Callable[[float], float],
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        cast_key = self.storage.cast_key
        rows: Iterable[Any] = (await conn.execute(stmt)).all()
        cb: dict[K, str] = {}
        sm: dict[K, float] = {}
        for k, v, s in rows:
            kk = cast_key(k)
            cb[kk] = v
            sm[kk] = score_fn(float(s))
        return cb, sm


@dataclass(slots=True)
class pgvector[K: int | str](
    RetrieverFunc[K, str, float],
    RrfMixin,
):
    """Sync facade over :class:`pgvector_async` retriever.

    Wraps a sync :class:`cbrkit.indexable.pgvector` storage.
    """

    storage: pgvector_storage[K, Any]
    search_type: Literal["dense", "sparse", "hybrid"] | None = None
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: Filter | None = None
    hybrid_oversample: int = 4
    normalize_scores: bool = True

    def _build_inner(self) -> pgvector_async[K]:
        """Build a fresh async retriever from the current field values.

        Constructed per call so mutations to the sync wrapper's fields
        (``limit``, ``where``, ...) take effect on subsequent
        invocations.
        """
        return pgvector_async[K](
            storage=self.storage.async_storage,
            search_type=self.search_type,
            query_conversion_func=self.query_conversion_func,
            limit=self.limit,
            where=self.where,
            hybrid_oversample=self.hybrid_oversample,
            normalize_scores=self.normalize_scores,
            rrf_k=self.rrf_k,
            rrf_weights=self.rrf_weights,
        )

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        return run_coroutine(self._build_inner()(batches))


__all__ = ["pgvector_async", "pgvector"]
