"""PostgreSQL/pgvector storage backend (async-first, with sync facade).

Two flavors live here:

- :class:`postgresql_async` is the primary, async-first implementation
  satisfying :class:`cbrkit.typing.AsyncFilterableIndexableFunc`.  Hosts
  may bring their own :class:`sqlalchemy.ext.asyncio.AsyncEngine` and/or
  :class:`sqlalchemy.Table` (set ``manage_schema=False`` to opt out of
  DDL), letting Alembic-managed projects (such as hivegent) keep the
  vector table inside their own ``Base.metadata``.

- :class:`postgresql` is a thin sync facade over :class:`postgresql_async`
  satisfying :class:`cbrkit.typing.FilterableIndexableFunc`.  It only
  accepts a connection URL and runs the async methods via
  :func:`cbrkit.helpers.run_coroutine`.

- :func:`postgresql_table` is a module-level factory that returns a
  :class:`sqlalchemy.Table` shaped for this backend.  It registers the
  table on the caller's :class:`sqlalchemy.MetaData` so Alembic
  autogenerate picks it up.
"""

from collections.abc import (
    AsyncIterator,
    Callable,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import TSVECTOR, insert as pg_insert
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    create_async_engine,
)

from ..filter import And, Eq, Filter, In, Like, Not, Or, Raw
from ..helpers import get_logger, run_coroutine
from ..typing import (
    BatchConversionFunc,
    Casebase,
    NumpyArray,
)
from ._common import (
    PG_METRICS,
    _compute_index_diff,
    _normalize_patch_keys,
)

logger = get_logger(__name__)


def _compile_filter(table: sa.Table, f: Filter) -> sa.ColumnElement[bool]:
    """Compile a backend-agnostic :class:`Filter` to a SQLAlchemy expression."""
    if isinstance(f, Eq):
        return table.c[f.column] == f.value
    if isinstance(f, In):
        return table.c[f.column].in_(list(f.values))
    if isinstance(f, Like):
        col = table.c[f.column]
        if f.escape is not None:
            return col.like(f.pattern, escape=f.escape)
        return col.like(f.pattern)
    if isinstance(f, And):
        return sa.and_(*(_compile_filter(table, x) for x in f.filters))
    if isinstance(f, Or):
        return sa.or_(*(_compile_filter(table, x) for x in f.filters))
    if isinstance(f, Not):
        return sa.not_(_compile_filter(table, f.inner))
    if isinstance(f, Raw):
        # ``sa.text`` returns a ``TextClause`` which SQLAlchemy accepts
        # anywhere a boolean column expression is required.
        return cast("sa.ColumnElement[bool]", cast(object, sa.text(f.sql)))
    raise TypeError(f"Unsupported filter node: {type(f).__name__}")


def postgresql_table(
    name: str,
    *,
    metadata: sa.MetaData,
    vector_dim: int,
    index_type: Literal["dense", "sparse", "hybrid"] = "dense",
    key_type: Literal["int", "str"] = "str",
    key_column: str = "key",
    value_column: str = "value",
    vector_column: str = "vector",
    tsv_column: str = "tsv",
    text_search_config: str = "english",
    metadata_columns: Mapping[str, sa.types.TypeEngine[Any]] | None = None,
) -> sa.Table:
    """Build a :class:`sqlalchemy.Table` shaped for the PostgreSQL backend.

    Registers the table on the supplied :paramref:`metadata` so Alembic
    autogenerate can discover it.  All column names and types are explicit
    — no runtime inference.

    Args:
        name: Table name within the database.
        metadata: Caller's :class:`sqlalchemy.MetaData` (typically
            ``Base.metadata``).
        vector_dim: Required when *index_type* is ``"dense"`` or
            ``"hybrid"`` — must match the embedding model's dimension.
        index_type: Determines which columns are added.
        key_type: ``"int"`` for ``BigInteger`` primary key, ``"str"`` for
            ``Text``.
        key_column / value_column / vector_column / tsv_column:
            Override default column names.
        text_search_config: PostgreSQL FTS configuration name used in
            the persisted ``tsvector`` generated column expression.
        metadata_columns: Mapping of column name to SQLAlchemy type for
            extra application metadata columns (e.g. ``casebase_key``).

    Returns:
        A :class:`sqlalchemy.Table` instance registered on *metadata*.
    """
    if index_type in ("dense", "hybrid") and vector_dim <= 0:
        raise ValueError(
            f"vector_dim must be positive for index_type={index_type!r}"
        )

    key_sa: sa.types.TypeEngine[Any] = (
        sa.BigInteger() if key_type == "int" else sa.Text()
    )
    columns: list[sa.Column[Any]] = [
        sa.Column(key_column, key_sa, primary_key=True),
        sa.Column(value_column, sa.Text(), nullable=False),
    ]

    if index_type in ("dense", "hybrid"):
        columns.append(
            sa.Column(vector_column, Vector(vector_dim), nullable=False)
        )

    if index_type in ("sparse", "hybrid"):
        columns.append(
            sa.Column(
                tsv_column,
                TSVECTOR(),
                sa.Computed(
                    sa.func.to_tsvector(
                        sa.literal(text_search_config),
                        sa.column(value_column),
                    ),
                    persisted=True,
                ),
                nullable=False,
            )
        )

    if metadata_columns:
        for col_name, col_type in metadata_columns.items():
            columns.append(sa.Column(col_name, col_type, nullable=True))

    return sa.Table(name, metadata, *columns)


@dataclass(slots=True)
class postgresql_async[K: int | str]:
    """Async-first PostgreSQL/pgvector storage.

    Implements :class:`cbrkit.typing.AsyncFilterableIndexableFunc`.

    Exactly one of *url* or *engine* must be supplied.  When the host
    passes a *table*, cbrkit treats the schema as host-managed
    (``manage_schema=False`` is implied — the table's column types are
    used as-is, no DDL is run).  When neither *table* nor *metadata* is
    passed, cbrkit builds its own :class:`sa.MetaData` and creates the
    table lazily on the first :meth:`put_index` call.

    Args:
        url: SQLAlchemy async URL (e.g.
            ``postgresql+psycopg://user:pw@host/db``).
        engine: A pre-built :class:`AsyncEngine`.  Mutually exclusive
            with *url*.
        table: Pre-declared :class:`sa.Table`.  When set, cbrkit does
            not touch DDL — :meth:`put_index` only writes rows.
        metadata: :class:`sa.MetaData` to register the table on when
            cbrkit builds the table itself.  Defaults to a new
            ``MetaData``.
        table_name: Table name (ignored when *table* is given).
        manage_schema: When ``True`` (default) and *table* is not given,
            run ``CREATE EXTENSION IF NOT EXISTS vector`` and create
            the table + indices on first write.
        vector_dim: Required when ``manage_schema=True`` and
            *index_type* ∈ {``"dense"``, ``"hybrid"``}.
        index_type / key_type / column names: As described in
            :func:`postgresql_table`.
        metric_type: Distance metric used by the HNSW index (and
            mirrored at search time by the retriever wrapper).
        conversion_func: Embedding function.  Required for
            ``"dense"`` / ``"hybrid"`` index types.
        metadata_func: Optional callable returning extra column values
            per row from ``(key, value)``.  Output keys MUST be a
            subset of the columns declared on the table (either via
            *table* / *metadata_columns*).
        metadata_columns / metadata_indexes: Schema hints used only
            when cbrkit builds the table itself
            (``manage_schema=True`` without *table*).
    """

    url: str | None = None
    engine: AsyncEngine | None = None
    table: sa.Table | None = None
    metadata: sa.MetaData | None = None
    table_name: str = "cases"
    manage_schema: bool = True
    vector_dim: int | None = None
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    key_type: Literal["int", "str"] = "str"
    key_column: str = "key"
    value_column: str = "value"
    vector_column: str = "vector"
    tsv_column: str = "tsv"
    text_search_config: str = "english"
    metric_type: Literal["cosine", "ip", "l2"] = "cosine"
    conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    metadata_func: Callable[[K, str], Mapping[str, Any]] | None = None
    metadata_columns: Mapping[str, sa.types.TypeEngine[Any]] | None = None
    metadata_indexes: Sequence[str | tuple[str, ...]] | None = None
    _engine: AsyncEngine = field(init=False, repr=False)
    _owns_engine: bool = field(init=False, default=False, repr=False)
    _table: sa.Table = field(init=False, repr=False)
    _schema_ready: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if self.index_type in ("dense", "hybrid") and self.conversion_func is None:
            raise ValueError(
                f"conversion_func is required for index_type={self.index_type!r}"
            )
        if (self.url is None) == (self.engine is None):
            raise ValueError("Exactly one of url or engine must be set")

        if self.engine is not None:
            self._engine = self.engine
            self._owns_engine = False
        else:
            assert self.url is not None
            self._engine = create_async_engine(self.url)
            self._owns_engine = True

        if self.table is not None:
            self._table = self.table
            # Host owns DDL when a table is supplied.
            self.manage_schema = False
            self._schema_ready = True
            return

        if not self.manage_schema:
            raise ValueError(
                "manage_schema=False requires an explicit `table` "
                "argument (use postgresql_table(...) to build it)."
            )

        if self.index_type in ("dense", "hybrid") and self.vector_dim is None:
            raise ValueError(
                "vector_dim is required when manage_schema=True and "
                f"index_type={self.index_type!r}"
            )

        meta = self.metadata if self.metadata is not None else sa.MetaData()
        self._table = postgresql_table(
            self.table_name,
            metadata=meta,
            vector_dim=self.vector_dim or 0,
            index_type=self.index_type,
            key_type=self.key_type,
            key_column=self.key_column,
            value_column=self.value_column,
            vector_column=self.vector_column,
            tsv_column=self.tsv_column,
            text_search_config=self.text_search_config,
            metadata_columns=self.metadata_columns,
        )

    # -- schema setup ---------------------------------------------------------

    async def _ensure_schema(self, conn: AsyncConnection) -> None:
        """Create extension, table, and indices on first write (sync conn fn)."""
        if self._schema_ready or not self.manage_schema:
            self._schema_ready = True
            return

        def _create(sync_conn: sa.Connection) -> None:
            if self.index_type in ("dense", "hybrid"):
                sync_conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
            self._table.create(sync_conn, checkfirst=True)
            self._create_indices(sync_conn)

        await conn.run_sync(_create)
        self._schema_ready = True

    def _create_indices(self, conn: sa.Connection) -> None:
        if self.index_type in ("dense", "hybrid"):
            sa.Index(
                f"ix_{self._table.name}_{self.vector_column}",
                self._table.c[self.vector_column],
                postgresql_using="hnsw",
                postgresql_ops={
                    self.vector_column: PG_METRICS[self.metric_type].opclass
                },
            ).create(conn, checkfirst=True)

        if self.index_type in ("sparse", "hybrid"):
            sa.Index(
                f"ix_{self._table.name}_{self.tsv_column}",
                self._table.c[self.tsv_column],
                postgresql_using="gin",
            ).create(conn, checkfirst=True)

        for spec in self.metadata_indexes or ():
            cols = (spec,) if isinstance(spec, str) else tuple(spec)
            sa.Index(
                f"ix_{self._table.name}_{'_'.join(cols)}",
                *(self._table.c[c] for c in cols),
            ).create(conn, checkfirst=True)

    # -- helpers --------------------------------------------------------------

    def _cast_key(self, value: Any) -> K:
        return cast(K, int(value) if self.key_type == "int" else str(value))

    def _build_rows(self, data: Casebase[K, str]) -> list[dict[str, Any]]:
        keys = list(data.keys())
        values = list(data.values())

        rows: list[dict[str, Any]] = [
            {self.key_column: k, self.value_column: v}
            for k, v in zip(keys, values, strict=True)
        ]

        if self.index_type in ("dense", "hybrid"):
            assert self.conversion_func is not None
            for row, vec in zip(rows, self.conversion_func(values), strict=True):
                row[self.vector_column] = np.asarray(vec).tolist()

        if self.metadata_func is not None:
            for row, k, v in zip(rows, keys, values, strict=True):
                row.update(self.metadata_func(k, v))

        return rows

    def _upsert_stmt(self, rows: list[dict[str, Any]]) -> Any:
        stmt = pg_insert(self._table).values(rows)
        update_cols = [c for c in rows[0] if c != self.key_column]
        return stmt.on_conflict_do_update(
            index_elements=[self._table.c[self.key_column]],
            set_={c: stmt.excluded[c] for c in update_cols},
        )

    def _has_table(self, sync_conn: sa.Connection) -> bool:
        """Sync check whether the underlying table exists (use with ``run_sync``)."""
        return sa.inspect(sync_conn).has_table(self._table.name)

    # -- AsyncIndexableFunc interface ----------------------------------------

    async def has_index(self) -> bool:
        """Return whether the underlying table has been materialized."""
        async with self._engine.connect() as conn:
            return await conn.run_sync(self._has_table)

    async def get_index(self) -> Casebase[K, str]:
        """Return the full casebase from the table (empty if not materialized)."""
        async with self._engine.connect() as conn:
            if not await conn.run_sync(self._has_table):
                return {}
            kc = self._table.c[self.key_column]
            vc = self._table.c[self.value_column]
            rows = (await conn.execute(sa.select(kc, vc))).all()
        return {self._cast_key(k): v for k, v in rows}

    async def put_index(self, data: Casebase[K, str], /) -> None:
        """Synchronize the table with *data* (in-place, no truncate).

        Empty *data* is a no-op only when the table has not been
        materialized yet — we avoid running ``CREATE EXTENSION``/
        ``CREATE TABLE`` just to immediately DELETE from an empty table,
        but still clear an existing table when the local cache is cold.
        """
        if not data and not self._schema_ready:
            async with self._engine.connect() as conn:
                if not await conn.run_sync(self._has_table):
                    return

        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)

            if not data:
                await conn.execute(sa.delete(self._table))
                return

            existing = await self._read_all(conn)
            stale_keys, changed = _compute_index_diff(existing, data)

            if not stale_keys and not changed:
                return

            if stale_keys:
                await conn.execute(
                    sa.delete(self._table).where(
                        self._table.c[self.key_column].in_(list(stale_keys))
                    )
                )

            if changed:
                await conn.execute(self._upsert_stmt(self._build_rows(changed)))

    async def upsert_index(self, data: Casebase[K, str], /) -> None:
        if not data:
            return
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            await conn.execute(self._upsert_stmt(self._build_rows(data)))

    async def delete_index(self, keys: Collection[K], /) -> None:
        if not keys:
            return
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            await conn.execute(
                sa.delete(self._table).where(
                    self._table.c[self.key_column].in_(list(keys))
                )
            )

    async def patch_index(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        normalized = _normalize_patch_keys(upsert, delete)
        if normalized is None:
            return
        _, delete_keys = normalized

        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            if delete_keys:
                await conn.execute(
                    sa.delete(self._table).where(
                        self._table.c[self.key_column].in_(list(delete_keys))
                    )
                )
            if upsert:
                await conn.execute(self._upsert_stmt(self._build_rows(upsert)))

    # -- AsyncFilterableIndexableFunc interface ------------------------------

    async def keys_where(self, where: Filter, /) -> Collection[K]:
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    sa.select(self._table.c[self.key_column]).where(
                        _compile_filter(self._table, where)
                    )
                )
            ).all()
        return [self._cast_key(k) for (k,) in rows]

    async def delete_where(self, where: Filter, /) -> Collection[K]:
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            keys = list(await self._keys_where_conn(conn, where))
            if keys:
                await conn.execute(
                    sa.delete(self._table).where(_compile_filter(self._table, where))
                )
        return keys

    async def replace_where(
        self, where: Filter, data: Casebase[K, str], /
    ) -> Collection[K]:
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            old_keys = list(await self._keys_where_conn(conn, where))
            if old_keys:
                await conn.execute(
                    sa.delete(self._table).where(_compile_filter(self._table, where))
                )
            if data:
                await conn.execute(self._upsert_stmt(self._build_rows(data)))
        return old_keys

    async def _keys_where_conn(
        self, conn: AsyncConnection, where: Filter
    ) -> Iterable[K]:
        rows = (
            await conn.execute(
                sa.select(self._table.c[self.key_column]).where(
                    _compile_filter(self._table, where)
                )
            )
        ).all()
        return (self._cast_key(k) for (k,) in rows)

    async def _read_all(self, conn: AsyncConnection) -> Casebase[K, str]:
        kc = self._table.c[self.key_column]
        vc = self._table.c[self.value_column]
        rows = (await conn.execute(sa.select(kc, vc))).all()
        return {self._cast_key(k): v for k, v in rows}

    # -- re-embed helper -----------------------------------------------------

    async def reembed_all(self, batch_size: int = 1000) -> int:
        """Recompute embeddings for every row in place.

        Iterates pages of *batch_size* rows ordered by primary key,
        re-runs :paramref:`conversion_func` on the text values, and
        UPDATEs the vector column.  Metadata columns are not touched.

        Returns:
            The total number of rows updated.
        """
        if self.index_type not in ("dense", "hybrid"):
            return 0
        if self.conversion_func is None:
            raise ValueError("conversion_func is required for reembed_all")

        kc = self._table.c[self.key_column]
        vc = self._table.c[self.value_column]
        total = 0

        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)

            offset = 0
            while True:
                rows = (
                    await conn.execute(
                        sa.select(kc, vc).order_by(kc).limit(batch_size).offset(offset)
                    )
                ).all()
                if not rows:
                    break

                keys = [k for k, _ in rows]
                texts = [v for _, v in rows]
                vecs = self.conversion_func(texts)

                await conn.execute(
                    sa.update(self._table)
                    .where(self._table.c[self.key_column] == sa.bindparam("k"))
                    .values({self.vector_column: sa.bindparam("v")}),
                    [
                        {"k": k, "v": np.asarray(vec).tolist()}
                        for k, vec in zip(keys, vecs, strict=True)
                    ],
                )
                total += len(rows)
                offset += batch_size

        return total

    # -- lifecycle -----------------------------------------------------------

    async def stream_rows(
        self, batch_size: int = 1000
    ) -> AsyncIterator[Sequence[tuple[K, str]]]:
        """Yield ``(key, value)`` pages of size *batch_size*."""
        kc = self._table.c[self.key_column]
        vc = self._table.c[self.value_column]

        async with self._engine.connect() as conn:
            offset = 0
            while True:
                rows = (
                    await conn.execute(
                        sa.select(kc, vc).order_by(kc).limit(batch_size).offset(offset)
                    )
                ).all()
                if not rows:
                    return
                yield [(self._cast_key(k), v) for k, v in rows]
                offset += batch_size

    async def close(self) -> None:
        """Dispose of the engine when cbrkit owns it."""
        if self._owns_engine:
            await self._engine.dispose()


@dataclass(slots=True)
class postgresql[K: int | str]:
    """Sync facade over :class:`postgresql_async`.

    Implements :class:`cbrkit.typing.FilterableIndexableFunc`.  Accepts
    a URL only (no :class:`AsyncEngine`); internally creates and owns
    its async counterpart.
    """

    url: str
    table_name: str = "cases"
    manage_schema: bool = True
    vector_dim: int | None = None
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    key_type: Literal["int", "str"] = "str"
    key_column: str = "key"
    value_column: str = "value"
    vector_column: str = "vector"
    tsv_column: str = "tsv"
    text_search_config: str = "english"
    metric_type: Literal["cosine", "ip", "l2"] = "cosine"
    conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    metadata_func: Callable[[K, str], Mapping[str, Any]] | None = None
    metadata_columns: Mapping[str, sa.types.TypeEngine[Any]] | None = None
    metadata_indexes: Sequence[str | tuple[str, ...]] | None = None
    _inner: postgresql_async[K] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._inner = postgresql_async[K](
            url=self.url,
            table_name=self.table_name,
            manage_schema=self.manage_schema,
            vector_dim=self.vector_dim,
            index_type=self.index_type,
            key_type=self.key_type,
            key_column=self.key_column,
            value_column=self.value_column,
            vector_column=self.vector_column,
            tsv_column=self.tsv_column,
            text_search_config=self.text_search_config,
            metric_type=self.metric_type,
            conversion_func=self.conversion_func,
            metadata_func=self.metadata_func,
            metadata_columns=self.metadata_columns,
            metadata_indexes=self.metadata_indexes,
        )

    @property
    def index(self) -> Casebase[K, str]:
        return run_coroutine(self._inner.get_index())

    def has_index(self) -> bool:
        return run_coroutine(self._inner.has_index())

    def put_index(self, data: Casebase[K, str], /) -> None:
        run_coroutine(self._inner.put_index(data))

    def upsert_index(self, data: Casebase[K, str], /) -> None:
        run_coroutine(self._inner.upsert_index(data))

    def delete_index(self, keys: Collection[K], /) -> None:
        run_coroutine(self._inner.delete_index(keys))

    def patch_index(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        run_coroutine(self._inner.patch_index(upsert=upsert, delete=delete))

    def keys_where(self, where: Filter, /) -> Collection[K]:
        return run_coroutine(self._inner.keys_where(where))

    def delete_where(self, where: Filter, /) -> Collection[K]:
        return run_coroutine(self._inner.delete_where(where))

    def replace_where(self, where: Filter, data: Casebase[K, str], /) -> Collection[K]:
        return run_coroutine(self._inner.replace_where(where, data))

    def reembed_all(self, batch_size: int = 1000) -> int:
        return run_coroutine(self._inner.reembed_all(batch_size))

    def close(self) -> None:
        run_coroutine(self._inner.close())


__all__ = ["postgresql_async", "postgresql", "postgresql_table"]
