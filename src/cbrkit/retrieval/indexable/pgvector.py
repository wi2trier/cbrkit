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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, override

import numpy as np
import sqlalchemy as sa

from ...filter import Filter
from ...helpers import forward_fields, run_coroutine
from ...indexable import (
    pgvector as pgvector_storage,
    pgvector_async as pgvector_async_storage,
)
from ...indexable._common import PG_METRICS
from ...typing import (
    BatchConversionFunc,
    Casebase,
    NumpyArray,
    RetrieverFunc,
    SimMap,
)
from ._common import (
    RrfMixin,
    SqlAlchemyVectorRetriever,
    reciprocal_rank_fusion,
)


def _build_dense_stmt(
    storage: pgvector_async_storage[Any, Any],
    qvec: list[float],
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    assert storage.value_column is not None
    table = storage.sa_table
    metric = PG_METRICS[storage.metric_type]
    distance = getattr(table.c[storage.pgvector_column], metric.distance_attr)(qvec)
    stmt = sa.select(
        table.c[storage.key_column],
        table.c[storage.value_column],
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
    assert storage.value_column is not None
    table = storage.sa_table
    tsv = table.c[storage.tsvector_column]
    tsq = sa.func.plainto_tsquery(storage.tsvector_config, query)
    rank = sa.func.ts_rank(tsv, tsq)
    stmt = (
        sa.select(
            table.c[storage.key_column],
            table.c[storage.value_column],
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
    SqlAlchemyVectorRetriever[K, pgvector_async_storage[K, Any]]
):
    """Async retriever wrapper for :class:`cbrkit.indexable.pgvector_async`.

    Queries are strings, matched against the storage's ``value_column``
    via dense (HNSW), sparse (GIN/FTS), or hybrid (RRF) search.  The
    returned casebase contains the text-column values; for full row
    data, read from the storage separately.

    The dispatch skeleton is inherited from
    :class:`~cbrkit.retrieval.indexable._common.AsyncVectorStorageRetriever`;
    only the three ``_search_db_*`` leaves are implemented here.
    *search_type* defaults to ``None`` and resolves to the storage's
    ``index_type`` at call time (via the base's ``_effective_search_type``)
    — set it explicitly only to query a subset of what was indexed (e.g.
    ``"dense"`` on a hybrid index).

    Note:
        When a call carries multiple queries they are executed
        sequentially on a single connection, not fanned out
        concurrently.  A single ``AsyncConnection`` cannot multiplex
        statements, and pgvector is interactive-first (usually one query
        per call), so the sequential path is the common-case optimum.
        Concurrent execution would require bounded pooled concurrency (a
        semaphore sized to the connection pool) to avoid exhausting it;
        this is deferred until batch throughput is a measured need.  For
        the same reason, hybrid search awaits its per-query dense and
        sparse statements in turn rather than gathering them.
    """

    # -- internal search helpers --------------------------------------------

    @override
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

    @override
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

    @override
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

                results.append(self._finalize_rrf(scores, values))

        return results


@dataclass(slots=True)
class pgvector[K: int | str](
    RetrieverFunc[K, str, float],
    RrfMixin,
):
    """Sync facade over :class:`pgvector_async` retriever.

    Wraps a sync :class:`cbrkit.indexable.pgvector` storage.  Runs the
    async implementation via :func:`cbrkit.helpers.run_coroutine`, so the
    same sequential-query behavior documented on :class:`pgvector_async`
    applies here.
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
            **forward_fields(self, exclude={"storage"}),
        )

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        return run_coroutine(self._build_inner()(batches))


__all__ = ["pgvector_async", "pgvector"]
