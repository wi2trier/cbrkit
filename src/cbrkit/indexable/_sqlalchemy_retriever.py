"""Retriever base of the SQLAlchemy storages with a vector index."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, override

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .._util import fuse_rankings
from ..typing import Casebase, NumpyArray, Results, SimMap
from ._retriever import (
    Hybrid,
    VectorStorageRetriever,
    _VectorStorage,
)


def check_vectors(
    backend: str, value_column: str | None, vector_dim: int | None
) -> None:
    """Raises a `ValueError` if a SQLAlchemy storage with a vector index has no text column or an invalid dimension."""
    if value_column is None:
        raise ValueError(f"value_column is required for {backend}")

    if vector_dim is not None and vector_dim <= 0:
        raise ValueError(f"vector_dim must be a positive int (got {vector_dim!r})")


class _SQLAlchemyVectorStorage(_VectorStorage, Protocol):
    """SQLAlchemy storage that a retriever searches."""

    @property
    def engine(self) -> AsyncEngine: ...

    def _has_table(self, sync_conn: sa.Connection) -> bool: ...

    def _scoped(
        self, where: sa.ColumnElement[bool] | None = None
    ) -> sa.ColumnElement[bool]: ...


@dataclass(slots=True, frozen=True, kw_only=True)
class SQLAlchemyVectorRetriever[K, St: _SQLAlchemyVectorStorage](
    VectorStorageRetriever[K, St], ABC
):
    """Retriever of a SQLAlchemy storage, shared by the pgvector and sqlite-vec backends.

    Subclasses provide the statements of a dense and a sparse search and how their rows are scored,
    the retriever adds the filter and the limit and runs the queries of one call on one connection.

    Args:
        where: SQLAlchemy expression over the table or its model, added to every search
            together with the `scope` of the storage.
        oversample: Factor of the limit that each ranking of a hybrid search returns before the fusion,
            and that a filtered dense search of sqlite-vec fetches before the filter.
    """

    where: sa.ColumnElement[bool] | None = None
    oversample: int = 4

    @abstractmethod
    def _dense_stmt(self, qvec: NumpyArray, k: int | None) -> sa.Select[Any, Any, Any]:
        """Select the `(key, value, distance)` rows nearest to `qvec`, `k` candidates if the backend preselects them."""

    @abstractmethod
    def _sparse_stmt(self, query: str) -> sa.Select[Any, Any, Any]:
        """Select the `(key, value, score)` rows matching the full text of `query`."""

    @property
    @abstractmethod
    def _dense_score(self) -> Callable[[float], float]:
        """Similarity of a distance of `_dense_stmt`."""

    @property
    @abstractmethod
    def _sparse_score(self) -> Callable[[float], float]:
        """Similarity of a score of `_sparse_stmt`."""

    async def _dense_k(self, conn: AsyncConnection) -> int | None:
        """Candidates of a dense search, the limit by default."""
        return self.limit

    async def _hybrid_k(self, conn: AsyncConnection) -> int | None:
        """Candidates of each ranking of a hybrid search, the oversampled limit by default."""
        return self.limit * self.oversample if self.limit is not None else None

    async def _collect_rows(
        self,
        conn: AsyncConnection,
        stmt: sa.Select[Any, Any, Any],
        limit: int | None,
        score_fn: Callable[[float], float],
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        """Run `stmt` with the filter and `limit` and fold its `(key, value, score)` rows into a result pair."""
        stmt = stmt.where(self.storage._scoped(self.where))

        if limit is not None:
            stmt = stmt.limit(limit)

        cb: dict[K, str] = {}
        sm: dict[K, float] = {}

        for k, v, s in (await conn.execute(stmt)).all():
            cb[k] = v
            sm[k] = score_fn(float(s))

        return cb, sm

    @override
    async def _search_dense(self, queries: Sequence[str]) -> Results[K, str]:
        query_vecs = await self._embed(queries)
        score_fn = self._dense_score

        async with self.storage.engine.connect() as conn:
            if not await conn.run_sync(self.storage._has_table):
                return [({}, {}) for _ in queries]

            k = await self._dense_k(conn)

            return [
                await self._collect_rows(
                    conn, self._dense_stmt(qvec, k), self.limit, score_fn
                )
                for qvec in query_vecs
            ]

    @override
    async def _search_sparse(self, queries: Sequence[str]) -> Results[K, str]:
        score_fn = self._sparse_score

        async with self.storage.engine.connect() as conn:
            if not await conn.run_sync(self.storage._has_table):
                return [({}, {}) for _ in queries]

            return [
                await self._collect_rows(
                    conn, self._sparse_stmt(query), self.limit, score_fn
                )
                for query in queries
            ]

    @override
    async def _search_hybrid(
        self, queries: Sequence[str], search: Hybrid
    ) -> Results[K, str]:
        query_vecs = await self._embed(queries)
        dense_score, sparse_score = self._dense_score, self._sparse_score
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        async with self.storage.engine.connect() as conn:
            if not await conn.run_sync(self.storage._has_table):
                return [({}, {}) for _ in queries]

            k = await self._hybrid_k(conn)

            for query, qvec in zip(queries, query_vecs, strict=True):
                dense = await self._collect_rows(
                    conn, self._dense_stmt(qvec, k), k, dense_score
                )
                sparse = await self._collect_rows(
                    conn, self._sparse_stmt(query), k, sparse_score
                )
                results.append(
                    fuse_rankings([dense, sparse], search.weights, search.k, self.limit)
                )

        return results


__all__ = [
    "SQLAlchemyVectorRetriever",
    "check_vectors",
]
