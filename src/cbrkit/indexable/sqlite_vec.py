"""SQLite ``sqlite-vec`` tabular storage backend (async-first, sync facade).

Extends the dialect-agnostic :class:`cbrkit.indexable.sqlalchemy_async`
base with vector search powered by the `sqlite-vec
<https://github.com/asg017/sqlite-vec>`_ loadable extension.  The layout is
shaped to SQLite's own primitives rather than mirroring pgvector:

- **Dense** embeddings live in a ``vec0`` *virtual table* (``<table>_vec``)
  keyed by the casebase key, queried with ``sqlite-vec``'s KNN
  ``... MATCH ... AND k = N``.  Building on ``vec0`` (rather than a plain
  BLOB column) means the backend inherits future ``vec0`` capabilities —
  approximate nearest-neighbor indexing as it lands upstream — and supports
  quantized element types today via :paramref:`vector_type`.
- **Sparse** full-text search uses SQLite's built-in **FTS5** in a second
  shadow table (``<table>_fts``).

Both shadow tables are kept in sync with the main table: the FTS shadow
entirely by triggers, and the ``vec0`` shadow by an ``AFTER DELETE`` trigger
plus a Python-side insert on write (embeddings are computed in Python, so a
SQL trigger cannot populate them).  The ``sqlite-vec`` extension is loaded on
every connection via a SQLAlchemy ``connect`` event reaching the
``aiosqlite`` worker thread.

The shadow tables are cbrkit-owned auxiliary indexes, distinct from the
main data table: unlike pgvector (where the vector/FTS targets are *columns*
a host model can declare), a ``vec0`` virtual table cannot live inside a
SQLAlchemy model.  cbrkit therefore creates and maintains the shadows
*regardless* of :paramref:`manage_schema` — which governs only the main
data table.  A custom-schema app brings its own ``model`` / ``table`` and
calls :meth:`reindex` once to backfill the shadows from pre-existing rows;
data written through cbrkit afterwards stays in sync automatically.

Filtering composes with KNN by joining the ``vec0`` matches back to the main
table; because ``vec0`` returns a fixed ``k`` *before* the join, a ``where``
filter oversamples candidates (see the retriever's ``hybrid_oversample``)
and may under-fill the limit on highly selective filters.

The simplest setup stores plain strings (``V = str``) in a cbrkit-built
table::

    storage = sqlite_vec(
        url="sqlite+aiosqlite:///cases.db",
        value_column="text",
        vector_dim=384,
        index_type="hybrid",
        conversion_func=embed,
    )
    storage.put_index({"a": "red sedan"})  # index -> {"a": "red sedan"}
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import sqlite_vec as sqlite_vec_ext
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from ..helpers import forward_fields, run_coroutine
from ..typing import BatchConversionFunc, NumpyArray
from ._common import SQLITE_VEC_METRICS, SQLITE_VEC_TYPES
from .sqlalchemy import build_indexable_table, sqlalchemy, sqlalchemy_async


def _attach_sqlite_vec_loader(engine: AsyncEngine) -> None:
    """Load ``sqlite-vec`` on every new connection of a SQLite async engine.

    The real ``sqlite3`` connection lives in ``aiosqlite``'s worker thread,
    so the extension must be loaded *in that thread* — reached here through
    the async driver connection's coroutine API driven by the adapter's
    ``await_`` bridge.  Attaching is idempotent per engine.
    """
    sync_engine = engine.sync_engine
    if sync_engine.dialect.name != "sqlite":
        return
    if getattr(sync_engine, "_cbrkit_sqlite_vec_loaded", False):
        return
    setattr(sync_engine, "_cbrkit_sqlite_vec_loaded", True)  # noqa: B010
    ext_path = sqlite_vec_ext.loadable_path()

    @event.listens_for(sync_engine, "connect")
    def _load(dbapi_conn: Any, _: Any) -> None:
        driver = dbapi_conn.driver_connection
        dbapi_conn.await_(driver.enable_load_extension(True))
        dbapi_conn.await_(driver.load_extension(ext_path))
        dbapi_conn.await_(driver.enable_load_extension(False))


@dataclass(slots=True)
class sqlite_vec_async[K: int | str, V = Mapping[str, Any]](sqlalchemy_async[K, V]):
    """Async SQLite/``sqlite-vec`` tabular storage.

    Extends :class:`sqlalchemy_async` with a ``vec0`` shadow table for dense
    KNN and an FTS5 shadow table for sparse search, both derived from the
    inherited :paramref:`value_column` (the embeddable text column).

    Args:
        vector_column: Name of the vector column inside the ``vec0`` table.
        vector_dim: Embedding dimension.  Required for *index_type* ∈
            {``"dense"``, ``"hybrid"``} (``vec0`` declares the column as
            ``<type>[dim]``).
        vector_type: ``vec0`` element type — ``"float32"`` (exact) or
            ``"int8"`` (quantized, ~4x smaller; embeddings are quantized by
            ``sqlite-vec`` assuming unit-normalized vectors).
        index_type: ``"dense"`` (vector KNN only), ``"sparse"`` (FTS5 only),
            or ``"hybrid"`` (both).
        metric_type: Distance metric for the ``vec0`` table (and mirrored at
            search time by the retriever wrapper).
        fts_tokenizer: Optional FTS5 ``tokenize=`` directive (e.g.
            ``"porter unicode61"``); ``None`` uses the FTS5 default.
        conversion_func: Embedding function.  Required for ``"dense"`` /
            ``"hybrid"`` index types.
    """

    vector_column: str = "embedding"
    vector_dim: int | None = None
    vector_type: Literal["float32", "int8"] = "float32"
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    metric_type: Literal["cosine", "l2", "l1"] = "cosine"
    fts_tokenizer: str | None = None
    conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    _shadows_ready: bool = field(init=False, default=False, repr=False)

    @property
    def has_dense(self) -> bool:
        """Whether this storage maintains a dense ``vec0`` index."""
        return self.index_type in ("dense", "hybrid")

    @property
    def has_sparse(self) -> bool:
        """Whether this storage maintains a sparse FTS5 index."""
        return self.index_type in ("sparse", "hybrid")

    @property
    def vec_table_name(self) -> str:
        """Name of the ``vec0`` shadow table (``<table>_vec``)."""
        return f"{self.sa_table.name}_vec"

    @property
    def fts_table_name(self) -> str:
        """Name of the FTS5 shadow table (``<table>_fts``)."""
        return f"{self.sa_table.name}_fts"

    @property
    def vector_value_sql(self) -> str:
        """SQL template wrapping a bound float32 BLOB for the element type."""
        return SQLITE_VEC_TYPES[self.vector_type].value_template

    @property
    def fts_table(self) -> sa.Table:
        """A lightweight :class:`sa.Table` over the FTS5 shadow for queries.

        Lives on its own :class:`sa.MetaData` so it never participates in
        the main table's DDL; the retriever joins it back to the main table
        on the key column.
        """
        assert self.value_column is not None
        return build_indexable_table(
            self.fts_table_name,
            metadata=sa.MetaData(),
            key_column=self.key_column,
            key_type=self.key_type,
            columns={self.value_column: sa.Text()},
        )

    def _validate_init(self) -> None:
        super(sqlite_vec_async, self)._validate_init()
        if self.value_column is None:
            raise ValueError("value_column is required for sqlite_vec")
        if self.table is not None and self.value_column not in self.table.columns:
            raise ValueError(
                f"value_column={self.value_column!r} must be a column of the "
                "model / table"
            )
        if self.has_dense and self.conversion_func is None:
            raise ValueError(
                f"conversion_func is required for index_type={self.index_type!r}"
            )
        if self.has_dense and self.vector_dim is None:
            raise ValueError(
                f"vector_dim is required for index_type={self.index_type!r} "
                "(the vec0 shadow declares the column as <type>[dim])"
            )
        if self.vector_dim is not None and self.vector_dim <= 0:
            raise ValueError(
                f"vector_dim must be a positive int (got {self.vector_dim!r})"
            )

    def _init_engine(self) -> None:
        super(sqlite_vec_async, self)._init_engine()
        _attach_sqlite_vec_loader(self._engine)

    async def _ensure_schema(self, conn: AsyncConnection) -> None:
        # The base creates the main data table only when manage_schema=True;
        # the shadow indexes are cbrkit-owned and created unconditionally.
        await super(sqlite_vec_async, self)._ensure_schema(conn)
        await self._ensure_shadows(conn)

    async def _ensure_shadows(self, conn: AsyncConnection) -> None:
        if self._shadows_ready:
            return
        if self.has_dense or self.has_sparse:
            await conn.run_sync(self._create_shadows)
        self._shadows_ready = True

    def _create_shadows(self, sync_conn: sa.Connection) -> None:
        assert self.value_column is not None
        key, val, main = self.key_column, self.value_column, self._table.name

        if self.has_dense:
            assert self.vector_dim is not None
            vec = self.vec_table_name
            pk_type = "integer" if self.key_type == "int" else "text"
            col_type = SQLITE_VEC_TYPES[self.vector_type].column_type
            metric = SQLITE_VEC_METRICS[self.metric_type].distance_metric
            # vec0's own DDL parser does not accept quoted column identifiers.
            sync_conn.execute(
                sa.text(
                    f'CREATE VIRTUAL TABLE IF NOT EXISTS "{vec}" USING vec0('
                    f"{key} {pk_type} primary key, "
                    f"{self.vector_column} {col_type}[{self.vector_dim}] "
                    f"distance_metric={metric})"
                )
            )
            # The embedding is computed in Python, so only deletes are
            # trigger-maintained; inserts happen in _do_upsert.
            sync_conn.execute(
                sa.text(
                    f'CREATE TRIGGER IF NOT EXISTS "{vec}_ad" AFTER DELETE ON "{main}" '
                    f'BEGIN DELETE FROM "{vec}" WHERE "{key}" = old."{key}"; END'
                )
            )

        if self.has_sparse:
            fts = self.fts_table_name
            tok = f", tokenize='{self.fts_tokenizer}'" if self.fts_tokenizer else ""
            sync_conn.execute(
                sa.text(
                    f'CREATE VIRTUAL TABLE IF NOT EXISTS "{fts}" '
                    f'USING fts5("{key}" UNINDEXED, "{val}"{tok})'
                )
            )
            # Triggers keep the FTS shadow in sync with every write path
            # (the base upserts via delete-then-insert; UPDATE is covered too).
            sync_conn.execute(
                sa.text(
                    f'CREATE TRIGGER IF NOT EXISTS "{fts}_ai" AFTER INSERT ON "{main}" '
                    f'BEGIN INSERT INTO "{fts}"("{key}", "{val}") '
                    f'VALUES (new."{key}", new."{val}"); END'
                )
            )
            sync_conn.execute(
                sa.text(
                    f'CREATE TRIGGER IF NOT EXISTS "{fts}_ad" AFTER DELETE ON "{main}" '
                    f'BEGIN DELETE FROM "{fts}" WHERE "{key}" = old."{key}"; END'
                )
            )
            sync_conn.execute(
                sa.text(
                    f'CREATE TRIGGER IF NOT EXISTS "{fts}_au" AFTER UPDATE ON "{main}" '
                    f'BEGIN DELETE FROM "{fts}" WHERE "{key}" = old."{key}"; '
                    f'INSERT INTO "{fts}"("{key}", "{val}") '
                    f'VALUES (new."{key}", new."{val}"); END'
                )
            )

    async def _do_upsert(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        # Write the main table (triggers drop stale vec0/fts rows and refresh
        # fts), then repopulate the vec0 shadow with freshly computed vectors.
        await super(sqlite_vec_async, self)._do_upsert(conn, rows)
        if self.has_dense and rows:
            assert self.value_column is not None
            await self._insert_vectors(
                conn,
                [row[self.key_column] for row in rows],
                [row[self.value_column] for row in rows],
            )

    async def _insert_vectors(
        self, conn: AsyncConnection, keys: list[Any], texts: list[Any]
    ) -> None:
        """Embed *texts* and insert the vectors into the ``vec0`` shadow."""
        assert self.conversion_func is not None
        vectors = self.conversion_func(texts)
        stmt = sa.text(
            f'INSERT INTO "{self.vec_table_name}"'
            f'("{self.key_column}", "{self.vector_column}") '
            f"VALUES (:key, {self.vector_value_sql.format(':vec')})"
        )
        params = [
            {
                "key": key,
                "vec": sqlite_vec_ext.serialize_float32(
                    np.asarray(vec, dtype=np.float32).tolist()
                ),
            }
            for key, vec in zip(keys, vectors, strict=True)
        ]
        batch_size = max(1, self._PARAM_LIMIT // 2)
        for start in range(0, len(params), batch_size):
            await conn.execute(stmt, params[start : start + batch_size])

    async def reindex(self, batch_size: int = 1000) -> int:
        """Rebuild the shadow indexes from the existing main-table rows.

        Clears the ``vec0`` / FTS5 shadows and repopulates them by streaming
        the main table.  Use this once after pointing the storage at a host
        table that already holds data (writes made *through* cbrkit keep the
        shadows in sync on their own).

        Returns:
            The number of rows indexed.
        """
        assert self.value_column is not None
        total = 0

        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            kc = self._table.c[self.key_column]
            vc = self._table.c[self.value_column]
            if self.has_dense:
                await conn.execute(sa.text(f'DELETE FROM "{self.vec_table_name}"'))
            if self.has_sparse:
                await conn.execute(sa.text(f'DELETE FROM "{self.fts_table_name}"'))

            offset = 0
            while True:
                rows = (
                    await conn.execute(
                        sa.select(kc, vc).order_by(kc).limit(batch_size).offset(offset)
                    )
                ).all()
                if not rows:
                    break
                await self._populate_shadows(
                    conn, [r[0] for r in rows], [r[1] for r in rows]
                )
                total += len(rows)
                offset += batch_size

        return total

    async def _populate_shadows(
        self, conn: AsyncConnection, keys: list[Any], texts: list[Any]
    ) -> None:
        """Insert ``(key, text)`` pairs directly into the FTS5 / ``vec0`` shadows.

        Used by :meth:`reindex` to backfill from existing rows; the normal
        write path keeps FTS in sync via triggers instead.
        """
        if self.has_sparse:
            await conn.execute(
                sa.text(
                    f'INSERT INTO "{self.fts_table_name}"'
                    f'("{self.key_column}", "{self.value_column}") '
                    f"VALUES (:key, :val)"
                ),
                [{"key": k, "val": t} for k, t in zip(keys, texts, strict=True)],
            )
        if self.has_dense:
            await self._insert_vectors(conn, keys, texts)


@dataclass(slots=True)
class sqlite_vec[K: int | str, V = Mapping[str, Any]](sqlalchemy[K, V]):
    """Sync facade over :class:`sqlite_vec_async`.

    Adds the ``sqlite-vec``-specific configuration on top of the
    :class:`cbrkit.indexable.sqlalchemy` sync facade and overrides
    :meth:`_build_async` to construct a :class:`sqlite_vec_async`.
    """

    vector_column: str = "embedding"
    vector_dim: int | None = None
    vector_type: Literal["float32", "int8"] = "float32"
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    metric_type: Literal["cosine", "l2", "l1"] = "cosine"
    fts_tokenizer: str | None = None
    conversion_func: BatchConversionFunc[str, NumpyArray] | None = None

    def _build_async(self) -> sqlite_vec_async[K, V]:
        return sqlite_vec_async[K, V](
            engine=self._engine, **forward_fields(self, exclude={"url"})
        )

    @property
    def async_storage(self) -> sqlite_vec_async[K, V]:
        """The wrapped async storage (used by sync retriever facades)."""
        return cast("sqlite_vec_async[K, V]", self._async)

    def reindex(self, batch_size: int = 1000) -> int:
        """Rebuild the shadow indexes from existing main-table rows."""
        return run_coroutine(self.async_storage.reindex(batch_size))


__all__ = [
    "sqlite_vec",
    "sqlite_vec_async",
]
