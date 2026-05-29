"""SQLite ``sqlite-vec`` retriever wrappers (sync + async).

These retrievers query strings against the embeddable ``value_column`` of a
:class:`cbrkit.indexable.sqlite_vec_async` storage.  They satisfy
:class:`cbrkit.typing.AsyncRetrieverFunc` / ``RetrieverFunc`` over ``str``
queries and return ``Casebase[K, str]`` — the text-column projection.

The three search leaves are shaped to SQLite's primitives rather than
mirroring pgvector:

- **dense** — ``vec0`` KNN (``... MATCH ... AND k = N``) joined back to the
  main table for the value text and any :class:`Filter` ``WHERE`` clause;
- **sparse** — FTS5 ``MATCH`` ranked by ``bm25``, joined back to the main
  table so the same :class:`Filter` applies;
- **hybrid** — Reciprocal Rank Fusion of the dense and sparse rankings.

Because ``vec0`` returns a fixed ``k`` *before* the join, a ``where`` filter
oversamples candidates (factor :paramref:`hybrid_oversample`) and may
under-fill the limit on highly selective filters.

The retriever is a pure query path: index maintenance lives on the storage
that owns the index (``storage.put_index(...)`` etc.).
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, override

import numpy as np
import sqlite_vec as sqlite_vec_ext
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from ...filter import Filter
from ...helpers import forward_fields, run_coroutine
from ...indexable import (
    sqlite_vec as sqlite_vec_storage,
    sqlite_vec_async as sqlite_vec_async_storage,
)
from ...indexable._common import SQLITE_VEC_METRICS
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


def _serialize(vec: NumpyArray) -> bytes:
    """Pack an embedding into the ``sqlite-vec`` ``float32`` BLOB format."""
    return sqlite_vec_ext.serialize_float32(np.asarray(vec, dtype=np.float32).tolist())


def _build_dense_stmt(
    storage: sqlite_vec_async_storage[Any, Any],
    qblob: bytes,
    where: Filter | None,
    k: int,
    outer_limit: int | None,
) -> sa.Select[Any]:
    assert storage.value_column is not None
    main = storage.sa_table
    match_expr = storage.vector_value_sql.format(":qvec")
    knn = (
        sa.text(
            f'SELECT "{storage.key_column}" AS knn_key, distance '
            f'FROM "{storage.vec_table_name}" '
            f'WHERE "{storage.vector_column}" MATCH {match_expr} AND k = :k'
        )
        .bindparams(
            sa.bindparam("qvec", qblob, type_=sa.LargeBinary),
            sa.bindparam("k", k),
        )
        .columns(sa.column("knn_key"), sa.column("distance"))
        .subquery("knn")
    )
    stmt = (
        sa.select(
            main.c[storage.key_column],
            main.c[storage.value_column],
            knn.c.distance,
        )
        .select_from(main.join(knn, main.c[storage.key_column] == knn.c.knn_key))
        .order_by(knn.c.distance)
    )
    if where is not None:
        stmt = stmt.where(storage.compile_filter(where))
    if outer_limit is not None:
        stmt = stmt.limit(outer_limit)
    return stmt


def _build_sparse_stmt(
    storage: sqlite_vec_async_storage[Any, Any],
    query: str,
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    assert storage.value_column is not None
    main = storage.sa_table
    fts = storage.fts_table
    score = sa.func.bm25(sa.literal_column(f'"{storage.fts_table_name}"')).label("score")
    stmt = (
        sa.select(
            main.c[storage.key_column],
            main.c[storage.value_column],
            score,
        )
        .select_from(
            fts.join(main, main.c[storage.key_column] == fts.c[storage.key_column])
        )
        .where(fts.c[storage.value_column].op("MATCH")(sa.bindparam("ftsq", query)))
        .order_by(sa.literal_column("rank"))
    )
    if where is not None:
        stmt = stmt.where(storage.compile_filter(where))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


@dataclass(slots=True)
class sqlite_vec_async[K: int | str](
    SqlAlchemyVectorRetriever[K, sqlite_vec_async_storage[K, Any]]
):
    """Async retriever wrapper for :class:`cbrkit.indexable.sqlite_vec_async`.

    Queries are strings, matched against the storage's ``value_column`` via
    dense (``vec0`` KNN), sparse (FTS5), or hybrid (RRF) search.  The returned
    casebase contains the text-column values; for full row data, read from
    the storage separately.

    The dispatch skeleton is inherited from
    :class:`~cbrkit.retrieval.indexable._common.AsyncVectorStorageRetriever`;
    only the three ``_search_db_*`` leaves are implemented here.
    *search_type* defaults to ``None`` and resolves to the storage's
    ``index_type`` at call time — set it explicitly only to query a subset of
    what was indexed (e.g. ``"dense"`` on a hybrid index).

    Note:
        Queries in a call run sequentially on a single connection: an
        ``aiosqlite`` connection cannot multiplex statements.
    """

    async def _vec_count(self, conn: AsyncConnection) -> int:
        total = (
            await conn.execute(
                sa.text(f'SELECT count(*) FROM "{self.storage.vec_table_name}"')
            )
        ).scalar()
        return max(1, int(total or 0))

    async def _dense_k(self, conn: AsyncConnection) -> int:
        """``vec0`` KNN ``k``: oversample when filtering, else the bare limit.

        With no limit, fall back to the full row count (``vec0`` requires a
        finite ``k``).
        """
        if self.limit is None:
            return await self._vec_count(conn)
        return self.limit * self.hybrid_oversample if self.where is not None else self.limit

    @override
    async def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage.conversion_func is not None
        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = await asyncio.to_thread(embed_func, list(queries))
        score_fn = SQLITE_VEC_METRICS[self.storage.metric_type].sim_from_distance
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.sa_engine.connect() as conn:
            k = await self._dense_k(conn)
            for qvec in query_vecs:
                stmt = _build_dense_stmt(
                    self.storage, _serialize(qvec), self.where, k, self.limit
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
                # bm25 is more-negative-is-better; flip the sign for a score.
                results.append(await self._collect_rows(conn, stmt, lambda s: -s))

        return results

    @override
    async def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage.conversion_func is not None
        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = await asyncio.to_thread(embed_func, list(queries))
        cast_key = self.storage.cast_key
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.sa_engine.connect() as conn:
            candidate_n = (
                self.limit * self.hybrid_oversample
                if self.limit is not None
                else await self._vec_count(conn)
            )
            for query, qvec in zip(queries, query_vecs, strict=True):
                dense_rows = (
                    await conn.execute(
                        _build_dense_stmt(
                            self.storage, _serialize(qvec), self.where, candidate_n, None
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
class sqlite_vec[K: int | str](
    RetrieverFunc[K, str, float],
    RrfMixin,
):
    """Sync facade over :class:`sqlite_vec_async` retriever.

    Wraps a sync :class:`cbrkit.indexable.sqlite_vec` storage and runs the
    async implementation via :func:`cbrkit.helpers.run_coroutine`.
    """

    storage: sqlite_vec_storage[K, Any]
    search_type: Literal["dense", "sparse", "hybrid"] | None = None
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: Filter | None = None
    hybrid_oversample: int = 4
    normalize_scores: bool = True

    def _build_inner(self) -> sqlite_vec_async[K]:
        """Build a fresh async retriever so field mutations take effect."""
        return sqlite_vec_async[K](
            storage=self.storage.async_storage,
            **forward_fields(self, exclude={"storage"}),
        )

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        return run_coroutine(self._build_inner()(batches))


__all__ = ["sqlite_vec_async", "sqlite_vec"]
