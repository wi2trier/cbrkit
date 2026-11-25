"""SQLite storage backend with the sqlite-vec extension."""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from itertools import batched
from typing import Any, Literal, override
from weakref import WeakSet

import numpy as np
import sqlalchemy as sa
import sqlite_vec as sqlite_vec_ext
from aiosqlite import Connection as AiosqliteConnection
from sqlalchemy import Engine, event
from sqlalchemy.engine import AdaptedConnection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine
from sqlalchemy.pool import ConnectionPoolEntry, PoolProxiedConnection

from ..helpers import dist2sim
from ..sim.embed.metrics import Cosine, Euclidean, Manhattan
from ..typing import EmbeddingFunc, NumpyArray
from ._retriever import (
    IndexType,
    check_embedder,
    dense_embedder,
    embed_rows,
    has_dense,
    has_sparse,
)
from ._sqlalchemy import SQLAlchemy, build_indexable_table
from ._sqlalchemy_retriever import SQLAlchemyVectorRetriever, check_vectors


@dataclass(slots=True, frozen=True)
class _SQLiteVecMetric:
    """Per-metric `vec0` configuration."""

    distance_metric: str
    """`distance_metric=` keyword used in the `vec0` table definition."""
    sim_from_distance: Callable[[float], float]
    """Convert a `vec0` `distance` value to a similarity score."""


type SQLiteVecMetric = Cosine | Euclidean | Manhattan
"""Vector metric of a sqlite-vec index, one of `cbrkit.sim.embed.metrics.VECTOR_METRICS`."""

_SQLITE_VEC_METRICS: Mapping[type[SQLiteVecMetric], _SQLiteVecMetric] = {
    Cosine: _SQLiteVecMetric("cosine", dist2sim),
    Euclidean: _SQLiteVecMetric("l2", dist2sim),
    Manhattan: _SQLiteVecMetric("l1", dist2sim),
}


@dataclass(slots=True, frozen=True)
class _SQLiteVecType:
    """Per-element-type `vec0` configuration.

    `value_template` wraps a bound float32-BLOB placeholder into the SQL
    expression `vec0` expects for the declared column type, used verbatim
    for both inserts and the KNN `MATCH` query, so storage and query stay in
    lock-step.  Embeddings are always serialized as float32, the extension
    performs any quantization at the SQL layer.
    """

    column_type: str
    """`vec0` element type, combined with `[dim]` (e.g. `float` -> `float[384]`)."""
    value_template: str
    """SQL template wrapping a placeholder (`{0}`), e.g. `vec_f32({0})`."""


_SQLITE_VEC_TYPES: Mapping[Literal["float32", "int8"], _SQLiteVecType] = {
    "float32": _SQLiteVecType("float", "vec_f32({0})"),
    "int8": _SQLiteVecType("int8", "vec_quantize_int8(vec_f32({0}), 'unit')"),
}

# Row-dict key carrying the serialized vector from _populate_system_columns to
# _insert_vectors.  Deliberately not a column name: the vec0 shadow's columns
# are an independent namespace from the data table's, so keying by
# vector_column would make an unrelated host column of the same name collide.
_VECTOR_KEY = "__cbrkit_vector__"

# Engines whose checkout hook is already attached, tracked weakly so a
# discarded engine does not keep its entry alive.
_hooked_engines: WeakSet[Engine] = WeakSet()


def serialize_vector(vec: NumpyArray) -> bytes:
    """Pack an embedding into the `sqlite-vec` `float32` BLOB format."""
    return sqlite_vec_ext.serialize_float32(np.asarray(vec, dtype=np.float32).tolist())


def _attach_sqlite_vec_loader(engine: AsyncEngine) -> None:
    """Load `sqlite-vec` on every connection of a SQLite async engine.

    The real `sqlite3` connection lives in `aiosqlite`'s worker thread,
    so the extension must be loaded in that thread through the adapter's
    `run_async` method.

    Hooked on `checkout` rather than `connect` so connections a host engine
    opened before handing it over get loaded too, instead of failing with `no such module: vec0`.
    The `record.info` latch keeps the load to once per
    real connection, and attaching is idempotent per engine.
    """
    sync_engine = engine.sync_engine

    if sync_engine.dialect.name != "sqlite" or sync_engine in _hooked_engines:
        return

    _hooked_engines.add(sync_engine)
    ext_path = sqlite_vec_ext.loadable_path()

    async def _load_extension(driver: AiosqliteConnection) -> None:
        await driver.enable_load_extension(True)

        try:
            await driver.load_extension(ext_path)
        finally:
            await driver.enable_load_extension(False)

    @event.listens_for(sync_engine, "checkout")
    def _load(
        dbapi_conn: AdaptedConnection,
        record: ConnectionPoolEntry,
        _: PoolProxiedConnection,
    ) -> None:
        if record.info.get("cbrkit_sqlite_vec_loaded"):
            return

        dbapi_conn.run_async(_load_extension)
        record.info["cbrkit_sqlite_vec_loaded"] = True


def _delete(shadow: str, key: str) -> str:
    """The trigger statement that deletes the shadow row of a deleted main row."""
    return f'DELETE FROM "{shadow}" WHERE "{key}" = old."{key}";'


@dataclass(slots=True, frozen=True, kw_only=True)
class SQLiteVec[K: Hashable, V = Mapping[str, Any]](SQLAlchemy[K, V]):
    """SQLite/`sqlite-vec` tabular storage.

    Extends `SQLAlchemy` with a `vec0` shadow table for dense
    KNN and an FTS5 shadow table for sparse search, both derived from the
    inherited `value_column` (the embeddable text column).
    The layout follows SQLite's own primitives rather than mirroring pgvector:

    - **Dense** embeddings live in a `vec0` *virtual table* (`<table>_vec`)
      keyed by the casebase key, queried with the KNN `... MATCH ... AND k = N`.
      Building on `vec0` means the backend inherits future capabilities such as
      approximate nearest-neighbor indexing and supports quantized element types
      via `vector_type`.
    - **Sparse** full-text search uses SQLite's built-in **FTS5** in a second
      shadow table (`<table>_fts`).

    Triggers keep the FTS shadow in sync with the main table, and the `vec0`
    shadow is filled on write in Python (where the embeddings are computed) and
    cleaned by an `AFTER DELETE` trigger.
    The extension is loaded on every connection of the engine, hooked once at construction without I/O.
    Unlike the pgvector columns, a `vec0` virtual table cannot live inside a
    SQLAlchemy model, so cbrkit maintains the shadows regardless of
    `manage_schema`, which only governs the main table.
    An application calls `reconcile` before the storage serves,
    which fits the shadows to their declaration and backfills them from existing rows.

    Filters compose with KNN by joining the `vec0` matches back to the main
    table, and since `vec0` selects a fixed `k` before the join, a filtered
    search oversamples candidates and may under-fill the limit on highly
    selective filters.

    Example:
        storage = SQLiteVec(
            create_async_engine("sqlite+aiosqlite:///cases.db"),
            value_column="text",
            vector_dim=384,
            index_type="hybrid",
            embedder=embed,
            scope=Unscoped.TABLE,
        )
        await storage.create()
        await storage.put_index({"a": "red sedan"})

    Args:
        vector_column: Name of the vector column inside the `vec0` table.
        vector_dim: Embedding dimension.  Required for `index_type` ∈
            {`"dense"`, `"hybrid"`} (`vec0` declares the column as
            `<type>[dim]`).
        vector_type: `vec0` element type, `"float32"` (exact) or
            `"int8"` (quantized, ~4x smaller, embeddings are quantized by
            `sqlite-vec` assuming unit-normalized vectors).
        index_type: `"dense"` (vector KNN only), `"sparse"` (FTS5 only),
            or `"hybrid"` (both).
        metric: Distance metric for the `vec0` table (and mirrored at
            search time by the retriever).
        fts_tokenizer: Optional FTS5 `tokenize=` directive (e.g.
            `"porter unicode61"`), `None` uses the FTS5 default.
        embedder: Embedding function.  Required for `"dense"` /
            `"hybrid"` index types.
    """

    vector_column: str = "vector"
    vector_dim: int | None = None
    vector_type: Literal["float32", "int8"] = "float32"
    index_type: IndexType = "dense"
    metric: SQLiteVecMetric = field(default_factory=Cosine)
    fts_tokenizer: str | None = None
    embedder: EmbeddingFunc[str, NumpyArray] | None = None

    def __post_init__(self) -> None:
        super(SQLiteVec, self).__post_init__()
        _attach_sqlite_vec_loader(self.engine)

    @property
    def vec_table_name(self) -> str:
        """Name of the `vec0` shadow table (`<table>_vec`)."""
        return f"{self.sa_table.name}_vec"

    @property
    def fts_table_name(self) -> str:
        """Name of the FTS5 shadow table (`<table>_fts`)."""
        return f"{self.sa_table.name}_fts"

    @property
    def fingerprint_table_name(self) -> str:
        """Name of the table (`<table>_fingerprint`) that records what the shadows were built from, see `reconcile`."""
        return f"{self.sa_table.name}_fingerprint"

    @property
    def vector_value_sql(self) -> str:
        """SQL template wrapping a bound float32 BLOB for the element type."""
        return _SQLITE_VEC_TYPES[self.vector_type].value_template

    @property
    def fts_table(self) -> sa.Table:
        """A lightweight `sa.Table` over the FTS5 shadow for queries.

        Lives on its own `sa.MetaData` so it never participates in
        the main table's DDL, the retriever joins it back to the main table
        on the key column.
        """
        assert self.value_column is not None
        return build_indexable_table(
            self.fts_table_name,
            metadata=sa.MetaData(),
            key_column=self.key,
            key_type=self._table.c[self.key].type,
            columns={self.value_column: sa.Text()},
        )

    def _validate_init(self) -> None:
        super(SQLiteVec, self)._validate_init()
        check_vectors("sqlite_vec", self.value_column, self.vector_dim)
        check_embedder(self.index_type, self.embedder)

        if has_dense(self.index_type) and self.vector_dim is None:
            raise ValueError(
                f"vector_dim is required for index_type={self.index_type!r} "
                "(the vec0 shadow declares the column as <type>[dim])"
            )

    def _create_schema(self, sync_conn: sa.Connection) -> None:
        # The base creates the main data table only when manage_schema=True,
        # the shadow indexes are cbrkit-owned and created either way.
        super(SQLiteVec, self)._create_schema(sync_conn)
        self._create_shadows(sync_conn)

    def _shadows(self) -> list[tuple[str, str, str]]:
        """The kind, name, and definition of each schema object that cbrkit owns beside the main table.

        SQLite stores each as `CREATE <kind> "<name>" <definition>`, which `reconcile` compares.
        """
        assert self.value_column is not None
        key, val, main = self.key, self.value_column, self._table.name
        shadows = [
            ("TABLE", self.fingerprint_table_name, "(fingerprint TEXT NOT NULL)")
        ]

        if has_dense(self.index_type):
            assert self.vector_dim is not None
            vec = self.vec_table_name
            pk_type = (
                "integer" if isinstance(self._table.c[key].type, sa.Integer) else "text"
            )
            col_type = _SQLITE_VEC_TYPES[self.vector_type].column_type
            metric = _SQLITE_VEC_METRICS[type(self.metric)].distance_metric
            # vec0's own DDL parser does not accept quoted column identifiers.
            columns = f"{key} {pk_type} primary key, {self.vector_column} {col_type}[{self.vector_dim}]"
            # The embedding is computed in Python, so only deletes are
            # trigger-maintained, upserts happen in _do_upsert.
            shadows += [
                (
                    "VIRTUAL TABLE",
                    vec,
                    f"USING vec0({columns} distance_metric={metric})",
                ),
                (
                    "TRIGGER",
                    f"{vec}_ad",
                    f'AFTER DELETE ON "{main}" BEGIN {_delete(vec, key)} END',
                ),
            ]

        if has_sparse(self.index_type):
            fts = self.fts_table_name
            tok = f", tokenize='{self.fts_tokenizer}'" if self.fts_tokenizer else ""
            insert = f'INSERT INTO "{fts}"("{key}", "{val}") VALUES (new."{key}", new."{val}");'
            delete = _delete(fts, key)
            # Triggers keep the FTS shadow in sync with every write path,
            # including the base upsert that updates rows in place.
            shadows += [
                ("VIRTUAL TABLE", fts, f'USING fts5("{key}" UNINDEXED, "{val}"{tok})'),
                (
                    "TRIGGER",
                    f"{fts}_ai",
                    f'AFTER INSERT ON "{main}" BEGIN {insert} END',
                ),
                (
                    "TRIGGER",
                    f"{fts}_ad",
                    f'AFTER DELETE ON "{main}" BEGIN {delete} END',
                ),
                (
                    "TRIGGER",
                    f"{fts}_au",
                    f'AFTER UPDATE ON "{main}" BEGIN {delete} {insert} END',
                ),
            ]

        return shadows

    def _create_shadows(self, sync_conn: sa.Connection) -> None:
        for kind, name, definition in self._shadows():
            sync_conn.execute(
                sa.text(f'CREATE {kind} IF NOT EXISTS "{name}" {definition}')
            )

    async def _populate_system_columns(self, rows: list[dict[str, Any]]) -> None:
        """Embed the text column into a serialized vector on each row.

        The vector belongs to the `vec0` shadow rather than the main table, so
        it rides along under `_VECTOR_KEY` and the base narrows it back
        out (see `_table_rows`).
        """
        assert self.value_column is not None

        if (embedder := dense_embedder(self.index_type, self.embedder)) is not None:
            await embed_rows(
                embedder, rows, self.value_column, _VECTOR_KEY, serialize_vector
            )

    async def _do_upsert(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        # The main table upserts in place and triggers refresh fts, but vec0
        # supports no upsert, so the vectors of the rows are replaced here.
        await super(SQLiteVec, self)._do_upsert(conn, rows)

        if has_dense(self.index_type):
            await conn.execute(
                sa.text(
                    f'DELETE FROM "{self.vec_table_name}" WHERE "{self.key}" = :key'
                ).bindparams(self._vec_key()),
                [{"key": row[self.key]} for row in rows],
            )
            await self._insert_vectors(conn, rows)

    def _vec_key(self) -> sa.BindParameter[Any]:
        """The key parameter of a `vec0` statement.

        It binds through the key column's type, so the shadow stores
        the same spelling as the main table (e.g. hex for sa.Uuid).
        """
        return sa.bindparam("key", type_=self._table.c[self.key].type)

    async def _insert_vectors(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """Insert the vectors prepared on `rows` into the `vec0` shadow."""
        stmt = sa.text(
            f'INSERT INTO "{self.vec_table_name}"'
            f'("{self.key}", "{self.vector_column}") '
            f"VALUES (:key, {self.vector_value_sql.format(':vec')})"
        ).bindparams(self._vec_key())
        params = [{"key": row[self.key], "vec": row[_VECTOR_KEY]} for row in rows]

        for chunk in batched(params, self._param_batch(2)):
            await conn.execute(stmt, chunk)

    @property
    def _reindexed(self) -> bool:
        """`reindex` rebuilds the shadows, which writes outside cbrkit leave behind."""
        return True

    @override
    async def reindex(self, batch_size: int = 1000, atomic: bool = True) -> int:
        """Rebuild the shadows in one transaction by default, keeping the old index on failure."""
        return await super(SQLiteVec, self).reindex(batch_size, atomic)

    async def reconcile(self, fingerprint: str, /) -> bool:
        """Fit the shadows to their declaration and rebuild them if they or what made the vectors changed.

        A shadow of another definition, such as of another dimension, metric, tokenizer, or column,
        is dropped with its recorded fingerprint and created again empty,
        and the fingerprint is recorded only after the rebuild, so a crash at any point makes the next call rebuild again.
        The rebuild runs in one transaction, see `reindex`.

        Args:
            fingerprint: What determines the stored vectors, such as the embedding model and its document prefix,
                but not the code that computes them, after a change of which the host calls `reindex` itself.

        Returns:
            Whether the shadows were rebuilt.

        Raises:
            Refused: When the storage is scoped, since a rebuild writes every row.
        """
        self._require_unscoped()

        async with self.engine.begin() as conn:
            if await conn.run_sync(self._fit_shadows) == fingerprint:
                return False

        _ = await self.reindex()
        table = self.fingerprint_table_name

        async with self.engine.begin() as conn:
            await conn.execute(sa.text(f'DELETE FROM "{table}"'))
            await conn.execute(
                sa.text(f'INSERT INTO "{table}" VALUES (:fingerprint)'),
                {"fingerprint": fingerprint},
            )

        return True

    def _fit_shadows(self, conn: sa.Connection) -> str | None:
        """Drop the shadows whose stored definition differs from the declared one, create the missing ones, and return the recorded fingerprint."""
        stored = {
            name: (kind, sql)
            for kind, name, sql in conn.execute(
                sa.text("SELECT type, name, sql FROM sqlite_schema")
            )
        }
        changed = False

        for kind, name, definition in self._shadows():
            if (live := stored.get(name)) is not None and live[1] != (
                f'CREATE {kind} "{name}" {definition}'
            ):
                conn.execute(sa.text(f'DROP {live[0].upper()} "{name}"'))
                changed = True

        self._create_shadows(conn)
        table = self.fingerprint_table_name

        if changed:
            conn.execute(sa.text(f'DELETE FROM "{table}"'))

            return None

        return conn.execute(sa.text(f'SELECT fingerprint FROM "{table}"')).scalar()

    async def _clear_index(self, conn: AsyncConnection) -> None:
        """Empty the `vec0` / FTS5 shadows before `reindex` fills them again."""
        if has_dense(self.index_type):
            await conn.execute(sa.text(f'DELETE FROM "{self.vec_table_name}"'))

        if has_sparse(self.index_type):
            await conn.execute(sa.text(f'DELETE FROM "{self.fts_table_name}"'))

    async def _write_index(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """Insert prepared rows directly into the FTS5 / `vec0` shadows.

        Used by `reindex` to backfill from existing rows, the normal
        write path keeps FTS in sync via triggers instead.
        """
        assert self.value_column is not None

        if has_sparse(self.index_type):
            await conn.execute(
                sa.insert(self.fts_table),
                [
                    {
                        self.key: row[self.key],
                        self.value_column: row[self.value_column],
                    }
                    for row in rows
                ],
            )
        if has_dense(self.index_type):
            await self._insert_vectors(conn, rows)


def _negate(score: float) -> float:
    """Turn a bm25 score, where more negative is better, into a similarity."""
    return -score


@dataclass(slots=True, frozen=True, kw_only=True)
class SQLiteVecRetriever[K: Hashable](SQLAlchemyVectorRetriever[K, SQLiteVec[K, Any]]):
    """Search of the `value_column` of a `SQLiteVec` storage.

    Several retrievers can share one storage to search it with different settings.
    The casebase of a result holds the text column, read full rows from the storage.
    The queries of one call run sequentially on a single connection.
    Since `vec0` selects its candidates before the `where` filter and the `scope` of the storage,
    a filtered dense search fetches `oversample` times the limit.
    """

    async def _vec_count(self, conn: AsyncConnection) -> int:
        total = (
            await conn.execute(
                sa.text(f'SELECT count(*) FROM "{self.storage.vec_table_name}"')
            )
        ).scalar()

        return max(1, int(total or 0))

    @override
    async def _dense_k(self, conn: AsyncConnection) -> int:
        """`vec0` KNN `k`: oversample when filtering or scoped, else the bare limit.

        With no limit, fall back to the full row count (`vec0` requires a
        finite `k`).
        """
        if self.limit is None:
            return await self._vec_count(conn)

        filtered = self.where is not None or self.storage._scope is not None

        return self.limit * self.oversample if filtered else self.limit

    @override
    async def _hybrid_k(self, conn: AsyncConnection) -> int:
        if self.limit is None:
            return await self._vec_count(conn)

        return self.limit * self.oversample

    @override
    def _dense_stmt(self, qvec: NumpyArray, k: int | None) -> sa.Select[Any, Any, Any]:
        storage = self.storage
        assert storage.value_column is not None
        main = storage.sa_table
        match_expr = storage.vector_value_sql.format(":qvec")
        knn = (
            sa.text(
                f'SELECT "{storage.key}" AS knn_key, distance '
                f'FROM "{storage.vec_table_name}" '
                f'WHERE "{storage.vector_column}" MATCH {match_expr} AND k = :k'
            )
            .bindparams(
                sa.bindparam("qvec", serialize_vector(qvec), type_=sa.LargeBinary),
                sa.bindparam("k", k),
            )
            .columns(
                sa.column("knn_key", main.c[storage.key].type),
                sa.column("distance"),
            )
            .subquery("knn")
        )

        return (
            sa.select(
                main.c[storage.key],
                main.c[storage.value_column],
                knn.c.distance,
            )
            .select_from(main.join(knn, main.c[storage.key] == knn.c.knn_key))
            .order_by(knn.c.distance)
        )

    @override
    def _sparse_stmt(self, query: str) -> sa.Select[Any, Any, Any]:
        storage = self.storage
        assert storage.value_column is not None
        main = storage.sa_table
        fts = storage.fts_table
        score = sa.func.bm25(sa.literal_column(f'"{storage.fts_table_name}"')).label(
            "score"
        )

        return (
            sa.select(
                main.c[storage.key],
                main.c[storage.value_column],
                score,
            )
            .select_from(fts.join(main, main.c[storage.key] == fts.c[storage.key]))
            .where(fts.c[storage.value_column].op("MATCH")(sa.bindparam("ftsq", query)))
            .order_by(sa.literal_column("rank"))
        )

    @property
    @override
    def _dense_score(self) -> Callable[[float], float]:
        return _SQLITE_VEC_METRICS[type(self.storage.metric)].sim_from_distance

    @property
    @override
    def _sparse_score(self) -> Callable[[float], float]:
        return _negate


__all__ = [
    "SQLiteVec",
    "SQLiteVecMetric",
    "SQLiteVecRetriever",
]
