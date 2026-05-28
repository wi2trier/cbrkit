"""PostgreSQL retriever wrappers (sync + async)."""

import asyncio
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection

from ...filter import Filter
from ...helpers import run_coroutine
from ...indexable import (
    PG_METRICS,
    postgresql as postgresql_storage,
    postgresql_async as postgresql_async_storage,
)
from ...indexable.postgresql import _compile_filter
from ...typing import (
    BatchConversionFunc,
    Casebase,
    NumpyArray,
    SimMap,
)
from ._common import (
    RrfMixin,
    _AsyncStorageIndexMixin,
    _brute_force_dense_search,
    _normalize_results,
    reciprocal_rank_fusion,
)


def _build_dense_stmt(
    storage: postgresql_async_storage[Any],
    qvec: list[float],
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    table = storage._table
    metric = PG_METRICS[storage.metric_type]
    distance = getattr(table.c[storage.vector_column], metric.distance_attr)(qvec)
    stmt = sa.select(
        table.c[storage.key_column],
        table.c[storage.value_column],
        distance.label("distance"),
    ).order_by(distance)
    if where is not None:
        stmt = stmt.where(_compile_filter(table, where))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


def _build_sparse_stmt(
    storage: postgresql_async_storage[Any],
    query: str,
    where: Filter | None,
    limit: int | None,
) -> sa.Select[Any]:
    table = storage._table
    tsv = table.c[storage.tsv_column]
    tsq = sa.func.plainto_tsquery(storage.text_search_config, query)
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
        stmt = stmt.where(_compile_filter(table, where))
    if limit is not None:
        stmt = stmt.limit(limit)
    return stmt


@dataclass(slots=True)
class postgresql_async[K: int | str](
    _AsyncStorageIndexMixin[K, postgresql_async_storage[K]],
    RrfMixin,
):
    """Async retriever wrapper for :class:`cbrkit.indexable.postgresql_async`.

    Index-maintenance methods come from :class:`_AsyncStorageIndexMixin`
    and RRF parameters from :class:`RrfMixin`.
    """

    storage: postgresql_async_storage[K]
    search_type: Literal["dense", "sparse", "hybrid"] = "dense"
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: Filter | None = None
    hybrid_oversample: int = 4
    normalize_scores: bool = True

    async def keys_where(self, where: Filter, /) -> Collection[K]:
        return await self.storage.keys_where(where)

    async def delete_where(self, where: Filter, /) -> Collection[K]:
        return await self.storage.delete_where(where)

    async def replace_where(
        self, where: Filter, data: Casebase[K, str], /
    ) -> Collection[K]:
        return await self.storage.replace_where(where, data)

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
                        "but no index is available. Call put_index() first."
                    )
                return (await self._search_db([query]))[0]
            return (await self._search_brute([query], casebase))[0]

        results = await asyncio.gather(
            *(per_batch(cb, q) for cb, q in batches)
        )
        return list(results)

    # -- internal search helpers --------------------------------------------

    async def _search_brute(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if self.search_type != "dense":
            raise NotImplementedError(
                f"Brute-force search is not supported for search_type={self.search_type!r}. "
                "Call put_index() first."
            )

        assert self.storage.conversion_func is not None
        results = await asyncio.to_thread(
            _brute_force_dense_search,
            queries,
            casebase,
            self.storage.conversion_func,
            self.query_conversion_func,
        )
        return cast(
            "Sequence[tuple[Casebase[K, str], SimMap[K, float]]]", results
        )

    async def _search_db(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        match self.search_type:
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

        async with self.storage._engine.connect() as conn:
            for qvec in query_vecs:
                stmt = _build_dense_stmt(
                    self.storage, np.asarray(qvec).tolist(), self.where, self.limit
                )
                results.append(
                    await self._collect_rows(conn, stmt, score_fn)
                )

        return results

    async def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage._engine.connect() as conn:
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
        cast_key = self.storage._cast_key
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage._engine.connect() as conn:
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
        cast_key = self.storage._cast_key
        rows: Iterable[Any] = (await conn.execute(stmt)).all()
        cb: dict[K, str] = {}
        sm: dict[K, float] = {}
        for k, v, s in rows:
            kk = cast_key(k)
            cb[kk] = v
            sm[kk] = score_fn(float(s))
        return cb, sm


@dataclass(slots=True)
class postgresql[K: int | str](RrfMixin):
    """Sync facade over :class:`postgresql_async` retriever.

    Wraps a sync :class:`cbrkit.indexable.postgresql` storage.  RRF
    parameters are inherited from :class:`RrfMixin`.
    """

    storage: postgresql_storage[K]
    search_type: Literal["dense", "sparse", "hybrid"] = "dense"
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: Filter | None = None
    hybrid_oversample: int = 4
    normalize_scores: bool = True

    def _build_inner(self) -> postgresql_async[K]:
        """Build a fresh async retriever from the current field values.

        Constructed per call so mutations to the sync wrapper's fields
        (`limit`, `where`, ...) take effect on subsequent invocations.
        """
        return postgresql_async[K](
            storage=self.storage._inner,
            search_type=self.search_type,
            query_conversion_func=self.query_conversion_func,
            limit=self.limit,
            where=self.where,
            hybrid_oversample=self.hybrid_oversample,
            normalize_scores=self.normalize_scores,
            rrf_k=self.rrf_k,
            rrf_weights=self.rrf_weights,
        )

    @property
    def index(self) -> Casebase[K, str]:
        return self.storage.index

    def has_index(self) -> bool:
        return self.storage.has_index()

    def put_index(self, data: Casebase[K, str], /) -> None:
        self.storage.put_index(data)

    def upsert_index(self, data: Casebase[K, str], /) -> None:
        self.storage.upsert_index(data)

    def delete_index(self, keys: Collection[K], /) -> None:
        self.storage.delete_index(keys)

    def patch_index(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        self.storage.patch_index(upsert=upsert, delete=delete)

    def keys_where(self, where: Filter, /) -> Collection[K]:
        return self.storage.keys_where(where)

    def delete_where(self, where: Filter, /) -> Collection[K]:
        return self.storage.delete_where(where)

    def replace_where(self, where: Filter, data: Casebase[K, str], /) -> Collection[K]:
        return self.storage.replace_where(where, data)

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        return run_coroutine(self._build_inner()(batches))


__all__ = ["postgresql_async", "postgresql"]
