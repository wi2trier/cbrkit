"""Dialect-agnostic SQLAlchemy storage backend (async-first, with sync facade).

The storage is *tabular*, and the value type ``V`` follows the schema
source: pass a SQLAlchemy mapped *model* for typed rows (``V`` = the model,
host-created table); a host *table* / ``reflect=True`` for ``Mapping``
rows; or just a *value_column* to let cbrkit build a one-column table with
``V = str``.  Backends that need additional columns (e.g. pgvector
``Vector`` / ``TSVECTOR``) subclass this module's :class:`sqlalchemy_async`
and declare them as *system columns* that are populated automatically on
write and hidden on read.

Two flavors live here:

- :class:`sqlalchemy_async` is the async-first implementation that
  satisfies :class:`cbrkit.typing.AsyncFilterableIndexableFunc`.  Hosts
  may bring their own :class:`sqlalchemy.ext.asyncio.AsyncEngine` and/or
  :class:`sqlalchemy.Table` (set ``manage_schema=False`` to opt out of
  DDL), letting Alembic-managed projects keep the table inside their own
  ``Base.metadata``.

- :class:`sqlalchemy` is a thin sync facade (URL-only,
  :class:`sqlalchemy.pool.NullPool`) over :class:`sqlalchemy_async`.  It
  satisfies :class:`cbrkit.typing.FilterableIndexableFunc`.  To read an
  existing table into a :class:`Casebase` without declaring its schema,
  pass ``reflect=True``:
  ``cbrkit.indexable.sqlalchemy(url, table_name=..., reflect=True).index``.

This is distinct from :class:`cbrkit.loaders.sqlalchemy`, which is a
one-shot read-only adapter that runs an *arbitrary query* into a
positionally-keyed mapping; the storage backend here owns a single table
keyed by a primary-key column and supports the full read/write/filter
interface.

Upserts use a portable delete-then-insert sequence; dialect subclasses
with native upsert (e.g. PostgreSQL ``ON CONFLICT``) override
:meth:`_do_upsert`.
"""

from collections.abc import (
    AsyncIterator,
    Collection,
    Iterable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, cast

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from ..filter import And, Eq, Filter, In, Like, Not, Or, Raw
from ..helpers import forward_fields, run_coroutine
from ..typing import (
    AsyncFilterableIndexableFunc,
    Casebase,
    FilterableIndexableFunc,
)
from ._common import RowCodec, _compute_index_diff, _normalize_patch_keys


def compile_filter(table: sa.Table, f: Filter) -> sa.ColumnElement[bool]:
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
        return sa.and_(*(compile_filter(table, x) for x in f.filters))
    if isinstance(f, Or):
        return sa.or_(*(compile_filter(table, x) for x in f.filters))
    if isinstance(f, Not):
        return sa.not_(compile_filter(table, f.inner))
    if isinstance(f, Raw):
        return cast("sa.ColumnElement[bool]", cast(object, sa.text(f.sql)))
    raise TypeError(f"Unsupported filter node: {type(f).__name__}")


def build_indexable_table(
    name: str,
    *,
    metadata: sa.MetaData,
    key_column: str = "key",
    key_type: Literal["int", "str"] = "str",
    columns: Mapping[str, sa.types.TypeEngine[Any]] = {},
    extra_columns: Sequence[sa.Column[Any]] = (),
) -> sa.Table:
    """Build a :class:`sqlalchemy.Table` with the standard indexable shape.

    ``columns`` are the user-facing payload columns (nullable).
    ``extra_columns`` carries backend-specific system columns (e.g.
    pgvector ``Vector`` / ``TSVECTOR``).
    """
    key_sa: sa.types.TypeEngine[Any] = (
        sa.BigInteger() if key_type == "int" else sa.Text()
    )
    sa_columns: list[sa.Column[Any]] = [
        sa.Column(key_column, key_sa, primary_key=True),
        *(sa.Column(c, t, nullable=True) for c, t in columns.items()),
        *extra_columns,
    ]
    return sa.Table(name, metadata, *sa_columns)


@dataclass(slots=True)
class sqlalchemy_async[K: int | str, V = Mapping[str, Any]](
    AsyncFilterableIndexableFunc[Casebase[K, V], Collection[K]]
):
    """Dialect-agnostic async tabular SQLAlchemy storage.

    Implements :class:`cbrkit.typing.AsyncFilterableIndexableFunc`.  The
    value type follows the schema source: a SQLAlchemy mapped *model*
    (typed rows), a host *table* / *reflect* (``Mapping`` rows), or just a
    *value_column* (``V = str``, cbrkit builds a one-column table).

    Exactly one of *url* or *engine* must be supplied.  When the host
    passes a *table* (or a *model*), cbrkit treats the schema as
    host-managed (``manage_schema=False`` is implied — the table is used
    as-is, no DDL is run).  Otherwise cbrkit builds its own
    :class:`sa.MetaData` and creates the (single-column) table lazily on
    the first write call.

    Args:
        url: SQLAlchemy async URL (e.g.
            ``sqlite+aiosqlite:///path/to.db``,
            ``postgresql+psycopg://user:pw@host/db``).
        engine: A pre-built :class:`AsyncEngine`.  Mutually exclusive with
            *url*.
        table: Pre-declared :class:`sa.Table`.  When set, cbrkit does not
            touch DDL — writes only INSERT/UPDATE/DELETE rows.  Rows are
            plain ``Mapping[str, Any]`` (``V = Mapping[str, Any]``).
        model: A SQLAlchemy mapped class (declarative, with ``__table__`` —
            optionally also a :class:`~sqlalchemy.orm.MappedAsDataclass` or
            Pydantic/SQLModel).  When set, ``V`` is the model type, cbrkit
            derives the *table* (``model.__table__``, host-created), and
            reads/writes rows as model instances.  The round-trip is driven
            off the resolved payload columns (key + system columns excluded):
            Pydantic models via ``model_dump`` / ``model_validate``, every
            other mapped class via ``getattr`` / its constructor.  Mutually
            exclusive with *table* and *reflect*.
        metadata: :class:`sa.MetaData` to register the table on when cbrkit
            builds it.  Defaults to a fresh ``MetaData``.
        table_name: Table name (ignored when *table* / *model* is given).
        manage_schema: When ``True`` (default) and *table* is not given,
            create the table + indices on first write.
        reflect: When ``True``, load the column schema from an *existing*
            database table by reflection on first use (implies
            ``manage_schema=False`` — cbrkit never issues DDL).  The key
            column and type are inferred from the reflected primary key when
            ``key_column`` is not present.  Mutually exclusive with a
            host-supplied *table*.
        key_column / key_type: Primary key column name and type
            (``"str"`` → ``Text``, ``"int"`` → ``BigInteger``).
        indexes: B-tree indices to create over payload columns
            (each entry is a column name or a tuple of column names).
        value_column: Names the single text column for the simplest schema:
            with no *model* / *table* / *reflect*, cbrkit builds a one-column
            table and ``V = str`` (the bare string is stored here and read
            back directly, so the value type matches what the text retrievers
            return).  Required in that case.

    Subclasses extend the schema with backend-specific *system columns*
    via :meth:`_system_columns` (vector, tsv, ...).  Those columns are
    hidden from reads (via :meth:`_system_column_names`) and populated on
    write by :meth:`_populate_system_columns`.
    """

    _PARAM_LIMIT: ClassVar[int] = 32_766
    """Per-statement bind parameter cap.

    SQLite caps at 32766 (since 3.32), PostgreSQL at 65535.  The
    conservative default works on both; subclasses with a known higher
    ceiling override.
    """

    url: str | None = None
    engine: AsyncEngine | None = None
    table: sa.Table | None = None
    model: type[V] | None = None
    metadata: sa.MetaData | None = None
    table_name: str = "cases"
    manage_schema: bool = True
    reflect: bool = False
    key_column: str = "key"
    key_type: Literal["int", "str"] = "str"
    indexes: Sequence[str | tuple[str, ...]] = ()
    value_column: str | None = None
    _engine: AsyncEngine = field(init=False, repr=False)
    _owns_engine: bool = field(init=False, default=False, repr=False)
    _table: sa.Table = field(init=False, repr=False)
    _reflect_meta: sa.MetaData = field(init=False, repr=False)
    _table_ready: bool = field(init=False, default=False, repr=False)
    _schema_ready: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._resolve_model()
        self._validate_init()
        self._init_engine()
        self._init_table()

    def _resolve_model(self) -> None:
        """Derive the table from a mapped *model* (dataclass or Pydantic).

        Runs before :meth:`_validate_init` so the derived table satisfies
        the host-managed-schema checks.  Row conversion is then handled by
        :attr:`_codec`, driven off the resolved table's payload columns.
        """
        if self.model is None:
            return
        if self.table is not None:
            raise ValueError("model and table are mutually exclusive")
        if self.reflect:
            raise ValueError("model and reflect are mutually exclusive")
        table = getattr(self.model, "__table__", None)
        if table is None:
            raise ValueError(
                f"model={self.model!r} must be a mapped ORM class with __table__"
            )
        self.table = cast(sa.Table, table)

    @property
    def _builds_own_schema(self) -> bool:
        """Whether cbrkit creates the table itself (str-mode), vs. host-managed."""
        return self.model is None and self.table is None and not self.reflect

    @property
    def _codec(self) -> RowCodec[V]:
        """Value <-> payload converter, scoped to cbrkit-owned columns.

        cbrkit-built schemas run in *str-mode* (``V = str``): the bare string
        lives in the single :attr:`value_column` and reads return it directly,
        mirroring the ``lancedb`` / ``zvec`` / ``chromadb`` default.  A *model*
        yields typed rows (Pydantic via ``model_dump`` / ``model_validate``,
        any other class — dataclass, SQLAlchemy mapped — via ``getattr`` /
        constructor); a host-supplied *table* / *reflect* yields ``Mapping``
        rows.  ``columns`` is the resolved payload column set (key + system
        columns excluded), so the round-trip is symmetric.
        """
        return RowCodec(
            model=self.model,
            columns=tuple(self._payload_column_names()),
            value_column=self.value_column if self._builds_own_schema else None,
        )

    # -- subclass hooks ------------------------------------------------------

    def _validate_init(self) -> None:
        """Validate constructor arguments. Subclasses extend (call super first)."""
        if (self.url is None) == (self.engine is None):
            raise ValueError("Exactly one of url or engine must be set")
        if self.reflect and self.table is not None:
            raise ValueError("reflect and table are mutually exclusive")
        if self._builds_own_schema and self.value_column is None:
            raise ValueError(
                "value_column is required when cbrkit builds the schema "
                "(pass a model, table, or reflect=True for richer rows)"
            )

    def _system_columns(self) -> Sequence[sa.Column[Any]]:
        """Column defs for backend-managed columns (vector, tsv, ...). Empty by default."""
        return ()

    def _system_column_names(self) -> frozenset[str]:
        """Names of system columns to hide on read. Empty by default."""
        return frozenset()

    def _create_system_indexes(self, sync_conn: sa.Connection) -> None:
        """DDL hook: create backend-specific indices (HNSW, GIN, ...). No-op by default."""
        return

    def _pre_create_ddl(self, sync_conn: sa.Connection) -> None:
        """DDL hook run before ``CREATE TABLE`` (e.g. ``CREATE EXTENSION``). No-op by default."""
        return

    def _populate_system_columns(self, rows: list[dict[str, Any]]) -> None:
        """Write hook: populate system columns on insert rows. No-op by default."""
        return

    async def _do_upsert(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """Upsert *rows* in parameter-cap-sized batches.

        Default: delete-then-insert per chunk (portable across every SA
        dialect).  Subclasses with native upsert (e.g. PostgreSQL
        ``ON CONFLICT``) override.
        """
        if not rows:
            return
        n_cols = len(rows[0])
        batch_size = max(1, self._PARAM_LIMIT // n_cols)
        kc = self._table.c[self.key_column]

        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            keys = [r[self.key_column] for r in chunk]
            await conn.execute(sa.delete(self._table).where(kc.in_(keys)))
            await conn.execute(sa.insert(self._table).values(chunk))

    # -- engine / table init -------------------------------------------------

    def _init_engine(self) -> None:
        if self.engine is not None:
            self._engine = self.engine
            self._owns_engine = False
            return
        assert self.url is not None
        self._engine = create_async_engine(self.url)
        self._owns_engine = True

    def _init_table(self) -> None:
        if self.table is not None:
            self._table = self.table
            self.manage_schema = False
            self._table_ready = True
            self._schema_ready = True
            return

        meta = self.metadata if self.metadata is not None else sa.MetaData()

        if self.reflect:
            self.manage_schema = False
            self._reflect_meta = meta
            return  # table is resolved lazily on first connection

        self._table = build_indexable_table(
            self.table_name,
            metadata=meta,
            key_column=self.key_column,
            key_type=self.key_type,
            columns=self._build_columns(),
            extra_columns=self._system_columns(),
        )
        self._table_ready = True

    def _build_columns(self) -> Mapping[str, sa.types.TypeEngine[Any]]:
        """Payload column spec for a cbrkit-built table (str-mode: one text column)."""
        assert self.value_column is not None
        return {self.value_column: sa.Text()}

    async def _ensure_table(self, conn: AsyncConnection) -> None:
        """Reflect the table from the database on first use (``reflect=True``)."""
        if self._table_ready:
            return

        def _reflect(sync_conn: sa.Connection) -> sa.Table:
            return sa.Table(
                self.table_name, self._reflect_meta, autoload_with=sync_conn
            )

        self._table = await conn.run_sync(_reflect)
        self._adopt_reflected_key()
        self._table_ready = True

    def _adopt_reflected_key(self) -> None:
        """Infer key column/type from the primary key when not already declared."""
        if self.key_column in self._table.c:
            return
        pk_cols = list(self._table.primary_key.columns)
        if len(pk_cols) != 1:
            raise ValueError(
                f"Cannot infer key column for reflected table {self.table_name!r}: "
                f"expected a single-column primary key, found {len(pk_cols)}. "
                "Pass key_column= explicitly."
            )
        pk = pk_cols[0]
        self.key_column = pk.name
        self.key_type = "int" if isinstance(pk.type, sa.Integer) else "str"

    # -- schema setup --------------------------------------------------------

    async def _ensure_schema(self, conn: AsyncConnection) -> None:
        await self._ensure_table(conn)
        if self._schema_ready or not self.manage_schema:
            self._schema_ready = True
            return

        def _create(sync_conn: sa.Connection) -> None:
            self._pre_create_ddl(sync_conn)
            self._table.create(sync_conn, checkfirst=True)
            self._create_system_indexes(sync_conn)
            self._create_payload_indexes(sync_conn)

        await conn.run_sync(_create)
        self._schema_ready = True

    def _create_payload_indexes(self, conn: sa.Connection) -> None:
        for spec in self.indexes:
            cols = (spec,) if isinstance(spec, str) else tuple(spec)
            sa.Index(
                f"ix_{self._table.name}_{'_'.join(cols)}",
                *(self._table.c[c] for c in cols),
            ).create(conn, checkfirst=True)

    # -- public hooks for retriever wrappers ---------------------------------

    @property
    def sa_engine(self) -> AsyncEngine:
        """The resolved async engine (whether host-supplied or cbrkit-owned)."""
        return self._engine

    @property
    def sa_table(self) -> sa.Table:
        """The resolved table (whether host-supplied or cbrkit-built)."""
        return self._table

    def cast_key(self, value: Any) -> K:
        """Cast a raw DB value to the configured key type."""
        return cast(K, int(value) if self.key_type == "int" else str(value))

    def compile_filter(self, where: Filter) -> sa.ColumnElement[bool]:
        """Compile a backend-agnostic :class:`Filter` against this table."""
        return compile_filter(self._table, where)

    # -- helpers -------------------------------------------------------------

    def _payload_column_names(self) -> list[str]:
        """Names of columns returned by reads (excludes key + system columns)."""
        system = self._system_column_names()
        return [
            c.name
            for c in self._table.columns
            if c.name != self.key_column and c.name not in system
        ]

    def _build_rows(self, data: Casebase[K, V]) -> list[dict[str, Any]]:
        codec = self._codec
        rows: list[dict[str, Any]] = []
        for k, v in data.items():
            payload = codec.encode(v)
            payload[self.key_column] = k
            rows.append(payload)
        self._populate_system_columns(rows)
        return rows

    async def _execute_delete_in(
        self, conn: AsyncConnection, keys: Iterable[K]
    ) -> None:
        keys_list = list(keys)
        if not keys_list:
            return
        kc = self._table.c[self.key_column]
        for start in range(0, len(keys_list), self._PARAM_LIMIT):
            chunk = keys_list[start : start + self._PARAM_LIMIT]
            await conn.execute(sa.delete(self._table).where(kc.in_(chunk)))

    def _has_table(self, sync_conn: sa.Connection) -> bool:
        name = self._table.name if self._table_ready else self.table_name
        schema = self._table.schema if self._table_ready else None
        return sa.inspect(sync_conn).has_table(name, schema=schema)

    async def _read_all(self, conn: AsyncConnection) -> dict[K, V]:
        codec = self._codec
        payload_names = self._payload_column_names()
        kc = self._table.c[self.key_column]
        payload_cols = [self._table.c[n] for n in payload_names]
        rows = (await conn.execute(sa.select(kc, *payload_cols))).all()
        return {
            self.cast_key(row[0]): codec.decode(
                dict(zip(payload_names, row[1:], strict=True))
            )
            for row in rows
        }

    async def _keys_where(self, conn: AsyncConnection, where: Filter) -> list[K]:
        rows = (
            await conn.execute(
                sa.select(self._table.c[self.key_column]).where(
                    self.compile_filter(where)
                )
            )
        ).all()
        return [self.cast_key(k) for (k,) in rows]

    # -- AsyncIndexableFunc interface ----------------------------------------

    async def has_index(self) -> bool:
        async with self._engine.connect() as conn:
            return await conn.run_sync(self._has_table)

    async def get_index(self) -> Casebase[K, V]:
        async with self._engine.connect() as conn:
            if not await conn.run_sync(self._has_table):
                return {}
            await self._ensure_table(conn)
            return await self._read_all(conn)

    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Synchronize the table with *data* (in-place, no truncate).

        Empty *data* is a no-op only when the table has not been
        materialized yet — we avoid running DDL just to immediately DELETE
        from an empty table, but still clear an existing table when the
        local cache is cold.
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
                await self._execute_delete_in(conn, stale_keys)
            if changed:
                await self._do_upsert(conn, self._build_rows(changed))

    async def upsert_index(self, data: Casebase[K, V], /) -> None:
        if not data:
            return
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            await self._do_upsert(conn, self._build_rows(data))

    async def delete_index(self, keys: Collection[K], /) -> None:
        if not keys:
            return
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            await self._execute_delete_in(conn, keys)

    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        normalized = _normalize_patch_keys(upsert, delete)
        if normalized is None:
            return
        _, delete_keys = normalized

        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            if delete_keys:
                await self._execute_delete_in(conn, delete_keys)
            if upsert:
                await self._do_upsert(conn, self._build_rows(upsert))

    # -- AsyncFilterableIndexableFunc interface ------------------------------

    async def keys_where(self, where: Filter, /) -> Collection[K]:
        async with self._engine.connect() as conn:
            await self._ensure_table(conn)
            return await self._keys_where(conn, where)

    async def delete_where(self, where: Filter, /) -> Collection[K]:
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            keys = await self._keys_where(conn, where)
            if keys:
                await conn.execute(
                    sa.delete(self._table).where(self.compile_filter(where))
                )
        return keys

    async def replace_where(
        self, where: Filter, data: Casebase[K, V], /
    ) -> Collection[K]:
        async with self._engine.begin() as conn:
            await self._ensure_schema(conn)
            old_keys = await self._keys_where(conn, where)
            if old_keys:
                await conn.execute(
                    sa.delete(self._table).where(self.compile_filter(where))
                )
            if data:
                await self._do_upsert(conn, self._build_rows(data))
        return old_keys

    # -- streaming / lifecycle -----------------------------------------------

    async def stream_rows(
        self, batch_size: int = 1000
    ) -> AsyncIterator[Sequence[tuple[K, V]]]:
        """Yield ``(key, row)`` pages of size *batch_size*."""
        async with self._engine.connect() as conn:
            await self._ensure_table(conn)
            codec = self._codec
            payload_names = self._payload_column_names()
            kc = self._table.c[self.key_column]
            payload_cols = [self._table.c[n] for n in payload_names]

            offset = 0
            while True:
                rows = (
                    await conn.execute(
                        sa.select(kc, *payload_cols)
                        .order_by(kc)
                        .limit(batch_size)
                        .offset(offset)
                    )
                ).all()
                if not rows:
                    return
                yield [
                    (
                        self.cast_key(row[0]),
                        codec.decode(dict(zip(payload_names, row[1:], strict=True))),
                    )
                    for row in rows
                ]
                offset += batch_size

    async def close(self) -> None:
        """Dispose of the engine when cbrkit owns it."""
        if self._owns_engine:
            await self._engine.dispose()


@dataclass(slots=True)
class sqlalchemy[K: int | str, V = Mapping[str, Any]](
    FilterableIndexableFunc[Casebase[K, V], Collection[K]]
):
    """Sync facade over :class:`sqlalchemy_async`.

    URL-only construction (no host-supplied engine or table) — the wrapped
    async engine uses :class:`sqlalchemy.pool.NullPool` so each call opens
    and closes its own connection, and nothing is pooled across the
    per-call event loops that :func:`cbrkit.helpers.run_coroutine` spins
    up.

    Unlike :class:`cbrkit.loaders.sqlalchemy` — a one-shot, read-only adapter
    that runs an arbitrary query into a positionally-keyed mapping — this is a
    *persistent, writable* store over a single table, keyed by a primary-key
    column, exposing upserts, deletes, and :class:`Filter`-based queries.

    Reading an existing table is the simplest case: pass ``reflect=True`` to
    derive the schema automatically, then read ``.index`` for a
    :class:`Casebase`::

        cbrkit.indexable.sqlalchemy(url, table_name="cases", reflect=True).index

    """

    url: str
    table_name: str = "cases"
    model: type[V] | None = None
    manage_schema: bool = True
    reflect: bool = False
    key_column: str = "key"
    key_type: Literal["int", "str"] = "str"
    indexes: Sequence[str | tuple[str, ...]] = ()
    value_column: str | None = None
    _engine: AsyncEngine = field(init=False, repr=False)
    _async: sqlalchemy_async[K, V] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._engine = create_async_engine(self.url, poolclass=NullPool)
        self._async = self._build_async()

    def _build_async(self) -> sqlalchemy_async[K, V]:
        return sqlalchemy_async[K, V](
            engine=self._engine, **forward_fields(self, exclude={"url"})
        )

    @property
    def async_storage(self) -> sqlalchemy_async[K, V]:
        """The wrapped async storage (used by sync retriever facades)."""
        return self._async

    @property
    def index(self) -> Casebase[K, V]:
        return run_coroutine(self._async.get_index())

    def has_index(self) -> bool:
        return run_coroutine(self._async.has_index())

    def put_index(self, data: Casebase[K, V], /) -> None:
        run_coroutine(self._async.put_index(data))

    def upsert_index(self, data: Casebase[K, V], /) -> None:
        run_coroutine(self._async.upsert_index(data))

    def delete_index(self, keys: Collection[K], /) -> None:
        run_coroutine(self._async.delete_index(keys))

    def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        run_coroutine(self._async.patch_index(upsert=upsert, delete=delete))

    def keys_where(self, where: Filter, /) -> Collection[K]:
        return run_coroutine(self._async.keys_where(where))

    def delete_where(self, where: Filter, /) -> Collection[K]:
        return run_coroutine(self._async.delete_where(where))

    def replace_where(self, where: Filter, data: Casebase[K, V], /) -> Collection[K]:
        return run_coroutine(self._async.replace_where(where, data))

    def close(self) -> None:
        run_coroutine(self._engine.dispose())


__all__ = [
    "build_indexable_table",
    "compile_filter",
    "sqlalchemy",
    "sqlalchemy_async",
]
