"""Dialect-agnostic SQLAlchemy storage backend."""

from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Collection,
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, field, replace
from enum import Enum
from inspect import signature, unwrap
from itertools import batched
from typing import Any, ClassVar, Self, cast
from uuid import UUID, uuid7

import sqlalchemy as sa
from sqlalchemy.dialects import mysql, postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from .._util import (
    check_expected,
    compute_index_diff,
    merge_patch,
    next_keys,
)
from ..errors import Conflict, Refused
from ..typing import Casebase, FilterableIndex, Storage
from ._common import RowCodec


def build_indexable_table(
    name: str,
    *,
    metadata: sa.MetaData,
    key_column: str = "key",
    key_type: sa.types.TypeEngine[Any],
    columns: Mapping[str, sa.types.TypeEngine[Any]] = {},
    extra_columns: Sequence[sa.Column[Any]] = (),
) -> sa.Table:
    """Build a `sqlalchemy.Table` with the standard indexable shape.

    `columns` are the user-facing payload columns (nullable).
    `extra_columns` carries backend-specific system columns (e.g.
    pgvector `Vector` / `TSVECTOR`).
    """
    sa_columns: list[sa.Column[Any]] = [
        sa.Column(key_column, key_type, primary_key=True),
        *(sa.Column(c, t, nullable=True) for c, t in columns.items()),
        *extra_columns,
    ]
    return sa.Table(name, metadata, *sa_columns)


async def reflect(
    engine: AsyncEngine,
    name: str,
    *,
    schema: str | None = None,
    metadata: sa.MetaData | None = None,
) -> sa.Table:
    """Read the columns and the primary key of an existing table once.

    The result is the `table` of a storage over a table that the host does not declare,
    whose rows are then plain mappings.

    Args:
        engine: Engine of the database.
        name: Name of the table.
        schema: Database schema of the table, the default one if None.
        metadata: `sa.MetaData` to register the table on, a fresh one if None.
    """
    target = metadata if metadata is not None else sa.MetaData()

    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda sync_conn: sa.Table(
                name, target, schema=schema, autoload_with=sync_conn
            )
        )


_NATIVE_UPSERT = frozenset({"postgresql", "sqlite", "mysql", "mariadb"})


class Unscoped(Enum):
    """The `scope` of a storage over every row of its table.

    Only the indexer of a table shared by many scopes holds one,
    so the storage of a tenant is never unscoped by omission.
    """

    TABLE = "table"


type StorageScope = sa.ColumnElement[bool] | Unscoped
"""The rows a `SQLAlchemy` storage reaches, an expression or `Unscoped.TABLE` for all of them."""


@dataclass(slots=True, frozen=True)
class SQLAlchemy[K: Hashable, V = Mapping[str, Any]](
    FilterableIndex[K, V, sa.ColumnElement[bool]], Storage[K, V]
):
    """Dialect-agnostic tabular SQLAlchemy storage over an async engine of the host.

    The value type follows the schema source: a SQLAlchemy mapped `model`
    (typed rows), a `value_column` (`V = str`, over a host `table` or a one-column table
    that cbrkit builds), or a host `table` alone (`Mapping` rows).
    Unlike `cbrkit.io.SQLQuery`, a read-only adapter of an arbitrary query,
    it is a writable store over a single table keyed by its primary key,
    whose `*_where` methods take SQLAlchemy expressions over `sa_table` or the `model`.

    The host creates and disposes the `engine`, and the storage does no I/O until it is used.
    `create` creates the schema objects that cbrkit owns, once before the first use.
    When the host passes a `table` (or a `model`), the table is the host's
    and only INSERT/UPDATE/DELETE statements touch it,
    so Alembic-managed projects keep it inside their own `Base.metadata`.
    An existing table that the host does not declare is read once with
    `reflect` and passed as `table`.

    Backends that need additional columns (e.g. the pgvector `Vector` and
    `TSVECTOR`) subclass this storage and declare them as *system columns*
    that are populated automatically on write and hidden on read.
    Upserts are native on PostgreSQL, SQLite, MySQL, and MariaDB and update existing rows in place,
    so the columns that a write leaves out keep their values and those with an `onupdate` get it.
    A callable `onupdate` must take no arguments and is evaluated once per statement,
    since a native upsert has no execution context, so a table whose callable reads one raises `TypeError`.
    Other dialects delete and insert the rows again instead, so their columns get their defaults again.
    The system columns of a subclass come from `_system_columns`,
    are hidden from reads by `_system_column_names`, and are populated on write
    by `_populate_system_columns`, which runs outside the write transaction, see `_write_transaction`.
    A subclass whose index `reindex` rebuilds sets `_reindexed`, clears the index with `_clear_index`, and writes it with `_write_index`.

    Args:
        engine: Async engine of the database, created and disposed by the host,
            such as `create_async_engine("sqlite+aiosqlite:///cases.db")`.
        table: Pre-declared `sa.Table` of the host.  Rows are
            plain `Mapping[str, Any]` (`V = Mapping[str, Any]`) unless `value_column` is given.
        model: A SQLAlchemy mapped class (declarative, with `__table__`,
            optionally also a `sqlalchemy.orm.MappedAsDataclass` or
            Pydantic/SQLModel).  When set, `V` is the model type, cbrkit
            derives the `table` (`model.__table__`, host-created), and
            reads/writes rows as model instances.  The round-trip is driven
            off the resolved payload columns (key and system columns excluded):
            Pydantic models via `model_dump` / `model_validate`, every
            other mapped class via `getattr` / its constructor.  Mutually
            exclusive with `table`.
        metadata: `sa.MetaData` to register the table on when cbrkit
            builds it.  Defaults to a fresh `MetaData`.
        table_name: Name of the table that cbrkit builds, ignored when `table` / `model` is given.
        manage_schema: When `True` (default) and neither `table` nor `model`
            is given, `create` creates the table and its indices.
        key_column: Primary key column, the single primary key column of a host `table` or `model`
            and `"key"` for a table that cbrkit builds if None, see `key`.
        key_type: SQLAlchemy type of the key column of a table that cbrkit builds,
            such as `sa.Text()` for `str`, `sa.BigInteger()` for `int` or `sa.Uuid()` for `uuid.UUID`.
            Keys are read back as the Python type of the key column, so a host table determines it.
        indexes: B-tree indices to create over payload columns
            (each entry is a column name or a tuple of column names).
        value_column: Text column of plain text values (`V = str`) unless a `model` is given,
            so the value type matches what the text retrievers return.
            Writes set only the key and this column, so new rows get the defaults of the others
            and replaced rows keep their values, see `_do_upsert`, and reads return its contents.
            Over a host `table` the scope may test the other columns,
            and with no `model` / `table` cbrkit builds a one-column table, whose column is `"value"` if None.
            With a `model`, it only names the text column of the model, such as the one that vector backends embed.
        scope: Required SQLAlchemy expression over the table or its model that restricts the storage to the rows matching it,
            so one table backs many storages, such as one per user, or `Unscoped.TABLE` for every row.
            Reads, deletes, replacements, and the searches of its retrievers only see rows in scope,
            and a write of a row that is or would end up outside the scope raises `cbrkit.errors.Refused`
            and writes nothing.
            `narrow` restricts a storage further and never widens it.
            Integer keys are still allocated over the whole table,
            and `reindex`, which rebuilds every row, refuses a scoped storage.
            Writers of different scopes still serialize on the table lock of PostgreSQL,
            and new rows must carry the columns that the scope tests, as values or defaults,
            while replaced rows keep theirs.

    Examples:
        >>> import asyncio
        >>> from sqlalchemy.ext.asyncio import create_async_engine
        >>> async def main() -> dict[str, str]:
        ...     engine = create_async_engine("sqlite+aiosqlite://")
        ...     storage = SQLAlchemy[str, str](engine, scope=Unscoped.TABLE)
        ...     await storage.create()
        ...     await storage.put_index({"a": "alpha"})
        ...     cases = dict(await storage.get_index())
        ...     await engine.dispose()
        ...     return cases
        >>> asyncio.run(main())
        {'a': 'alpha'}
    """

    _PARAM_LIMIT: ClassVar[int] = 32_766
    """Per-statement bind parameter cap.

    SQLite caps at 32766 (since 3.32), PostgreSQL at 65535.  The
    conservative default works on both, subclasses with a known higher
    ceiling override.
    """

    engine: AsyncEngine
    _: KW_ONLY
    table: sa.Table | None = None
    model: type[V] | None = None
    metadata: sa.MetaData | None = None
    table_name: str = "cases"
    manage_schema: bool = True
    key_column: str | None = None
    key_type: sa.types.TypeEngine[Any] = field(default_factory=sa.Text)
    indexes: Sequence[str | tuple[str, ...]] = ()
    value_column: str | None = None
    scope: StorageScope
    _table: sa.Table = field(init=False, repr=False)
    _key: str = field(init=False, repr=False)
    _codec: RowCodec[V] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        host_table = self._host_table()

        if self._builds_own_schema and self.value_column is None:
            object.__setattr__(self, "value_column", "value")

        self._validate_init()
        table = host_table if host_table is not None else self._build_table()

        if self.value_column is not None and self.value_column not in table.c:
            raise ValueError(
                f"value_column={self.value_column!r} must be a column of the "
                "model / table"
            )

        if self.engine.dialect.name in _NATIVE_UPSERT:
            for column in table.columns:
                default = column.onupdate

                if (
                    isinstance(default, sa.ColumnDefault)
                    and default.is_callable
                    and signature(default.arg).parameters
                ):
                    raise TypeError(
                        f"The onupdate of column {column.name!r} reads the execution context, "
                        "which native upserts lack, use a SQL expression such as sa.func.now() "
                        "or a callable without arguments instead"
                    )

        # derived once at construction, the storage is frozen afterwards
        object.__setattr__(self, "_table", table)
        object.__setattr__(self, "_key", self._primary_key(table))
        object.__setattr__(self, "_codec", self._build_codec())

    async def create(self) -> None:
        """Create the schema objects that cbrkit owns unless they exist.

        These are the table and its indices if cbrkit builds the table
        and the backend-specific objects such as the shadow tables of `SQLiteVec`.
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(self._create_schema)

    def _host_table(self) -> sa.Table | None:
        """The table of a host `table` or mapped `model`, None if cbrkit builds it.

        Row conversion is then handled by `_codec`, driven off the
        resolved table's payload columns.
        """
        if self.model is None:
            return self.table
        if self.table is not None:
            raise ValueError("model and table are mutually exclusive")
        table = getattr(self.model, "__table__", None)
        if table is None:
            raise ValueError(
                f"model={self.model!r} must be a mapped ORM class with __table__"
            )
        return cast(sa.Table, table)

    @property
    def _builds_own_schema(self) -> bool:
        """Whether cbrkit creates the table itself (str-mode), vs. host-managed."""
        return self.model is None and self.table is None

    @property
    def key(self) -> str:
        """Name of the primary key column."""
        return self._key

    def _build_codec(self) -> RowCodec[V]:
        """Value <-> payload converter, scoped to cbrkit-owned columns.

        A `model` yields typed rows (Pydantic via `model_dump` / `model_validate`,
        any other class via `getattr` / constructor) over the payload columns
        (key and system columns excluded), so the round-trip is symmetric.
        Otherwise a `value_column` yields plain text (`V = str`) stored in that column alone,
        mirroring the `lancedb` / `zvec` default, and a host `table` alone yields `Mapping` rows.
        """
        if self.model is None and self.value_column is not None:
            return RowCodec(
                columns=(self.value_column,), value_column=self.value_column
            )

        return RowCodec(
            model=self.model,
            columns=tuple(self._payload_column_names()),
            key_column=self.key,
        )

    # -- subclass hooks ------------------------------------------------------

    def _validate_init(self) -> None:
        """Validate constructor arguments. Subclasses extend (call super first)."""
        return

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
        """DDL hook run before `CREATE TABLE` (e.g. `CREATE EXTENSION`). No-op by default."""
        return

    async def _populate_system_columns(self, rows: list[dict[str, Any]]) -> None:
        """Write hook: populate system columns on insert rows. No-op by default."""
        return

    @property
    def _reindexed(self) -> bool:
        """Whether `reindex` has an index to rebuild, none by default."""
        return False

    async def _clear_index(self, conn: AsyncConnection) -> None:
        """Reindex hook: clear the index before `_write_index` rebuilds it. No-op by default."""
        return

    async def _write_index(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """Reindex hook: write the index of `rows`, whose system columns are populated. No-op by default."""
        return

    def _param_batch(self, per_row: int) -> int:
        """Rows per statement that stay under the dialect's bind-parameter cap."""
        return max(1, self._PARAM_LIMIT // per_row)

    def _table_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Narrow `rows` to keys that are actually columns of the table.

        `_populate_system_columns` may attach a value belonging to a
        backend's side structure rather than to the table (sqlite_vec's `vec0`
        shadow), so the statements below narrow rather than trusting the dicts.
        """
        if not rows or not (extras := rows[0].keys() - self._table.c.keys()):
            return rows

        return [{k: v for k, v in row.items() if k not in extras} for row in rows]

    async def _do_upsert(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """Upsert `rows` in parameter-cap-sized batches.

        PostgreSQL and SQLite use `INSERT ... ON CONFLICT DO UPDATE`, MySQL and MariaDB
        `INSERT ... ON DUPLICATE KEY UPDATE`, and other dialects delete and insert the rows again.
        A native upsert updates the written columns and applies the `onupdate` of the others,
        which SQLAlchemy leaves out of such statements, evaluating a callable once per statement.
        """
        rows = self._table_rows(rows)
        if not rows:
            return

        kc = self._table.c[self.key]

        for chunk in batched(rows, self._param_batch(len(rows[0]))):
            stmt = self._upsert(conn.dialect.name, chunk)

            if stmt is None:
                await conn.execute(
                    sa.delete(self._table).where(kc.in_([r[self.key] for r in chunk]))
                )
                stmt = sa.insert(self._table).values(chunk)

            await conn.execute(stmt)

    def _upsert(
        self, dialect: str, rows: Sequence[Mapping[str, Any]]
    ) -> sa.Insert | None:
        """Native upsert of `rows`, None if the dialect has none."""
        columns = rows[0].keys()
        updates = {
            column.name: unwrap(default.arg)() if default.is_callable else default.arg
            for column in self._table.columns
            if column.name not in columns
            and isinstance(default := column.onupdate, sa.ColumnDefault)
        }

        match dialect:
            case "postgresql" | "sqlite":
                insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
                stmt = insert(self._table).values(rows)
                new = stmt.excluded
                return stmt.on_conflict_do_update(
                    index_elements=[self._table.c[self.key]],
                    set_={c: new[c] for c in columns if c != self.key} | updates,
                )
            case "mysql" | "mariadb":
                stmt = mysql.insert(self._table).values(rows)
                new = stmt.inserted
                return stmt.on_duplicate_key_update(
                    {c: new[c] for c in columns if c != self.key} | updates
                )
            case _:
                return None

    # -- table init -----------------------------------------------------------

    def _build_table(self) -> sa.Table:
        """The table that cbrkit builds when the host declares none."""
        return build_indexable_table(
            self.table_name,
            metadata=self.metadata if self.metadata is not None else sa.MetaData(),
            key_column=self.key_column or "key",
            key_type=self.key_type,
            columns=self._build_columns(),
            extra_columns=self._system_columns(),
        )

    def _build_columns(self) -> Mapping[str, sa.types.TypeEngine[Any]]:
        """Payload column spec for a cbrkit-built table (str-mode: one text column)."""
        assert self.value_column is not None
        return {self.value_column: sa.Text()}

    def _primary_key(self, table: sa.Table) -> str:
        """The key column, by default the single primary key column of the table."""
        if self.key_column is not None:
            if self.key_column not in table.c:
                raise ValueError(
                    f"key_column={self.key_column!r} is no column of table {table.name!r}"
                )

            return self.key_column

        pk_cols = list(table.primary_key.columns)

        if len(pk_cols) != 1:
            raise ValueError(
                f"Cannot infer key column for table {table.name!r}: "
                f"expected a single-column primary key, found {len(pk_cols)}. "
                "Pass key_column= explicitly."
            )

        return pk_cols[0].name

    # -- schema setup --------------------------------------------------------

    def _create_schema(self, sync_conn: sa.Connection) -> None:
        """Create every schema object cbrkit owns, run by `create`.

        The data table is the host's unless cbrkit builds it with `manage_schema`,
        so it is created only then.  Subclasses extend this with objects that are
        cbrkit's either way (shadow tables and their triggers).
        """
        if not (self.manage_schema and self._builds_own_schema):
            return

        self._pre_create_ddl(sync_conn)
        self._table.create(sync_conn, checkfirst=True)
        self._create_system_indexes(sync_conn)
        self._create_payload_indexes(sync_conn)

    def _create_payload_indexes(self, conn: sa.Connection) -> None:
        for spec in self.indexes:
            cols = (spec,) if isinstance(spec, str) else tuple(spec)
            sa.Index(
                f"ix_{self._table.name}_{'_'.join(cols)}",
                *(self._table.c[c] for c in cols),
            ).create(conn, checkfirst=True)

    # -- public hooks for the retrievers --------------------------------------

    @property
    def sa_table(self) -> sa.Table:
        """The resolved table (whether host-supplied or cbrkit-built)."""
        return self._table

    # -- helpers -------------------------------------------------------------

    def _payload_column_names(self) -> list[str]:
        """Names of columns returned by reads (excludes key + system columns), see `_codec`."""
        system = self._system_column_names()
        return [
            c.name
            for c in self._table.columns
            if c.name != self.key and c.name not in system
        ]

    async def _build_rows(self, data: Casebase[K, V]) -> list[dict[str, Any]]:
        """Build insert rows and populate their system columns, such as the embeddings of pgvector."""
        codec = self._codec
        rows: list[dict[str, Any]] = []
        for k, v in data.items():
            payload = codec.encode(v)
            payload[self.key] = k
            rows.append(payload)
        await self._populate_system_columns(rows)

        return rows

    @asynccontextmanager
    async def _write_transaction(
        self, data: Casebase[K, V] | None = None
    ) -> AsyncGenerator[tuple[AsyncConnection, list[dict[str, Any]]]]:
        """Open a write transaction with the schema resolved and `data` encoded.

        Both steps are deliberately hoisted out of the transaction.
        `_build_rows` populates the system columns, which for pgvector
        means embedding the whole batch against a GPU or a remote API, easily
        seconds, and doing that inside the transaction keeps its pooled
        connection checked out for the full duration, on PostgreSQL the
        resulting `idle in transaction` session also holds whatever locks the
        preceding statements took and pins the xmin horizon, stalling vacuum
        cleanup database-wide.  Hoisting the DDL out too means a first write
        that fails can leave the empty table behind, which `checkfirst=True`
        makes harmless.

        Callers whose rows are a diff against the current contents pass no
        `data` and build inside the block, so that the read and the write share
        one transaction (see `put_index`).
        """
        rows = await self._build_rows(data) if data else []

        async with self.engine.begin() as conn:
            yield conn, rows

    def _keys_in(self, keys: Sequence[K]) -> Iterator[sa.ColumnElement[bool]]:
        """Predicates matching `keys`, split to stay under the parameter cap.

        Dialects that bind a whole array as one parameter override this.
        """
        kc = self._table.c[self.key]

        for chunk in batched(keys, self._param_batch(1)):
            yield kc.in_(chunk)

    @property
    def _scope(self) -> sa.ColumnElement[bool] | None:
        """The `scope` as an expression, None for every row."""
        return None if isinstance(self.scope, Unscoped) else self.scope

    def narrow(self, where: sa.ColumnElement[bool]) -> Self:
        """This storage restricted to its rows that also match `where`."""
        scope = where if self._scope is None else sa.and_(self._scope, where)

        return replace(self, scope=scope)

    def _require_unscoped(self) -> None:
        """Raise `Refused` unless the storage reaches every row, as a rebuild of the table does."""
        if self._scope is not None:
            raise Refused("Only an unscoped storage rebuilds every row of the table")

    def _scoped(
        self, where: sa.ColumnElement[bool] | None = None
    ) -> sa.ColumnElement[bool]:
        """`where` restricted to the `scope`, each optional."""
        return sa.and_(sa.true(), *(c for c in (where, self._scope) if c is not None))

    async def _delete(
        self, conn: AsyncConnection, where: sa.ColumnElement[bool] | None = None
    ) -> None:
        await conn.execute(sa.delete(self._table).where(self._scoped(where)))

    async def _execute_delete_in(
        self, conn: AsyncConnection, keys: Iterable[K]
    ) -> None:
        for where in self._keys_in(list(keys)):
            await self._delete(conn, where)

    async def _check_scope(self, conn: AsyncConnection, keys: Sequence[K]) -> None:
        """Raise `Refused` if a row of `keys` exists outside the `scope`."""
        if (scope := self._scope) is None:
            return

        outside = sa.func.sum(sa.case((scope, 0), else_=1))

        for where in self._keys_in(keys):
            stmt = sa.select(outside).select_from(self._table).where(where)

            if (await conn.execute(stmt)).scalar():
                raise Refused(
                    "A write must not touch rows outside the scope of the storage"
                )

    async def _write(self, conn: AsyncConnection, rows: list[dict[str, Any]]) -> None:
        """Upsert `rows`, refusing rows of other scopes and rows that leave the scope.

        Raising inside the write transaction rolls it back, so nothing is written.
        """
        keys = [row[self.key] for row in rows]
        await self._check_scope(conn, keys)
        await self._do_upsert(conn, rows)
        await self._check_scope(conn, keys)

    def _has_table(self, sync_conn: sa.Connection) -> bool:
        return sa.inspect(sync_conn).has_table(
            self._table.name, schema=self._table.schema
        )

    async def _read_all(
        self, conn: AsyncConnection, where: sa.ColumnElement[bool] | None = None
    ) -> dict[K, V]:
        codec = self._codec
        payload_names = codec.columns
        kc = self._table.c[self.key]
        payload_cols = [self._table.c[n] for n in payload_names]
        stmt = sa.select(kc, *payload_cols).where(self._scoped(where))
        rows = (await conn.execute(stmt)).all()
        return {
            row[0]: codec.decode(dict(zip(payload_names, row[1:], strict=True)), row[0])
            for row in rows
        }

    async def _keys_where(
        self, conn: AsyncConnection, where: sa.ColumnElement[bool]
    ) -> list[K]:
        rows = (
            await conn.execute(
                sa.select(self._table.c[self.key]).where(self._scoped(where))
            )
        ).all()
        return [k for (k,) in rows]

    # -- Index interface ----------------------------------------

    async def has_index(self) -> bool:
        async with self.engine.connect() as conn:
            return await conn.run_sync(self._has_table)

    async def get_index(self) -> Casebase[K, V]:
        async with self.engine.connect() as conn:
            if not await conn.run_sync(self._has_table):
                return {}

            return await self._read_all(conn)

    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Synchronize the table with `data` (in-place, no truncate).

        The rows are built `inside` the transaction that read the existing ones,
        since they are the diff against what the table currently holds: reading
        in one transaction and writing the diff in another would silently
        overwrite a concurrent writer in between.
        """
        async with self._write_transaction() as (conn, _):
            if not data:
                await self._delete(conn)
                return

            existing = await self._read_all(conn)
            stale_keys, changed = compute_index_diff(existing, data)
            if stale_keys:
                await self._execute_delete_in(conn, stale_keys)
            if changed:
                await self._write(conn, await self._build_rows(changed))

    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
        add: Sequence[V] = (),
        *,
        insert: Casebase[K, V] | None = None,
        expected: Casebase[K, V] | None = None,
    ) -> list[K]:
        """Insert, replace, delete, and add cases in one transaction, see `cbrkit.typing.Storage`.

        Integer keys continue after the largest key read in the same transaction,
        and the expected and inserted cases are compared with the rows read in it,
        which PostgreSQL serializes with a table lock that still admits readers
        and SQLite with its single writer.
        Other keys are new UUIDv7 values, as `uuid.UUID` for UUID columns and as strings otherwise.
        """
        written, delete_keys = merge_patch(upsert, insert, delete)
        if not written and not delete_keys and not add:
            return []

        placeholders = cast(Casebase[K, V], dict(enumerate(add)))
        added = await self._build_rows(placeholders) if add else []

        async with self._write_transaction(written) as (conn, rows):
            if expected is not None or insert:
                await self._lock_writers(conn)
            if expected is not None:
                check_expected(await self._read_all(conn), expected)
            if insert:
                await self._check_absent(conn, list(insert))

            keys = await self._allocate_keys(conn, len(added), written)
            for row, key in zip(added, keys, strict=True):
                row[self.key] = key
            if delete_keys:
                await self._execute_delete_in(conn, delete_keys)
            if rows or added:
                await self._write(conn, [*rows, *added])

        return keys

    async def _lock_writers(self, conn: AsyncConnection) -> None:
        """Locks the table against other writers until the transaction ends, on PostgreSQL."""
        if conn.dialect.name == "postgresql":
            table = conn.dialect.identifier_preparer.format_table(self._table)
            await conn.execute(
                sa.text(f"LOCK TABLE {table} IN SHARE ROW EXCLUSIVE MODE")
            )

    async def _check_absent(self, conn: AsyncConnection, keys: Sequence[K]) -> None:
        """Raise `Conflict` if a row in scope has one of `keys`."""
        kc = self._table.c[self.key]

        for where in self._keys_in(keys):
            stmt = sa.select(kc).where(self._scoped(where)).limit(1)

            if (key := (await conn.execute(stmt)).scalar()) is not None:
                raise Conflict(f"{key!r} exists already")

    async def _allocate_keys(
        self, conn: AsyncConnection, count: int, written: Collection[K]
    ) -> list[K]:
        """Keys for `count` new rows by the type of the key column, none of the `written` ones."""
        if not count:
            return []
        kc = self._table.c[self.key]
        try:
            python_type: type = kc.type.python_type
        except NotImplementedError:
            python_type = str
        if issubclass(python_type, UUID):
            return cast(list[K], [uuid7() for _ in range(count)])
        if not issubclass(python_type, int):
            return cast(list[K], [str(uuid7()) for _ in range(count)])
        await self._lock_writers(conn)
        last = (await conn.execute(sa.select(sa.func.max(kc)))).scalar()
        return next_keys([*written, *([] if last is None else [last])], count)

    # -- FilterableIndex interface ------------------------------

    async def keys_where(self, where: sa.ColumnElement[bool], /) -> Collection[K]:
        async with self.engine.connect() as conn:
            if not await conn.run_sync(self._has_table):
                return []

            return await self._keys_where(conn, where)

    async def get_where(self, where: sa.ColumnElement[bool], /) -> Casebase[K, V]:
        async with self.engine.connect() as conn:
            if not await conn.run_sync(self._has_table):
                return {}

            return await self._read_all(conn, where)

    async def delete_where(self, where: sa.ColumnElement[bool], /) -> Collection[K]:
        async with self._write_transaction() as (conn, _):
            keys = await self._keys_where(conn, where)
            if keys:
                await self._delete(conn, where)

        return keys

    async def replace_where(
        self, where: sa.ColumnElement[bool], data: Casebase[K, V], /
    ) -> Collection[K]:
        """Atomically replace the rows matched by `where` with `data`.

        Callers must serialize concurrent replacements of the same predicate.
        """
        async with self._write_transaction(data) as (conn, rows):
            old_keys = await self._keys_where(conn, where)
            if old_keys:
                await self._delete(conn, where)
            if rows:
                await self._write(conn, rows)

        return old_keys

    # -- streaming / lifecycle -----------------------------------------------

    async def _iter_pages(
        self,
        *columns: sa.ColumnElement[Any],
        where: sa.ColumnElement[bool] | None = None,
        batch_size: int = 1000,
    ) -> AsyncIterator[Sequence[sa.Row[*tuple[Any, ...]]]]:
        """Yield `(key, *columns)` pages of the rows matching `where` in primary-key order.

        Keyset pagination rather than OFFSET: pages stay stable when a
        concurrent writer inserts or deletes below the cursor, where OFFSET
        would skip or repeat rows, and each page seeks the primary key index
        instead of rescanning everything ahead of it.  Every page takes a
        connection of its own, so a slow consumer never holds one.
        """
        kc = self._table.c[self.key]
        page = sa.select(kc, *columns).order_by(kc).limit(batch_size)

        if where is not None:
            page = page.where(where)

        cursor: Any = None

        while True:
            stmt = page if cursor is None else page.where(kc > cursor)
            async with self.engine.connect() as conn:
                rows = (await conn.execute(stmt)).all()

            if not rows:
                return

            yield rows

            # A short page means the range above the cursor is exhausted, so
            # stopping here saves the round trip that would confirm it.
            if len(rows) < batch_size:
                return

            cursor = rows[-1][0]

    async def reindex(self, batch_size: int = 1000, atomic: bool = False) -> int:
        """Rebuild the index of every row from its value column, page by page.

        Walks the key and value columns, re-runs `_populate_system_columns`, such as the embedder,
        and writes the index of the populated rows, so it backfills the index of rows written outside cbrkit
        and refreshes it after the embedder changed, while the other columns stay untouched.
        The population stays outside any transaction whichever way `atomic` goes,
        for the reasons in `_write_transaction`, which over a whole table apply many times over.

        Args:
            batch_size: Rows read and indexed per page.
            atomic: Buffer the index of every row and write it in one transaction at the end,
                so an interrupted run changes nothing.
                Both of its costs scale with the table: the buffer holds every value with its index,
                and the closing transaction holds its connection for a write per row.
                A paged rebuild commits each page as it is produced, which keeps it feasible on a table of any size,
                but until it returns, a search mixes the old index with the new one,
                such as vectors of two models whose distances are meaningless.
                A rebuild before the storage serves never exposes that state and keeps the default.

        Returns:
            The number of rows walked, 0 if the storage keeps no index to rebuild.

        Raises:
            Refused: When the storage is scoped, since a rebuild writes every row.
        """
        self._require_unscoped()

        if not self._reindexed:
            return 0

        assert self.value_column is not None
        buffered: list[dict[str, Any]] = []
        total = 0

        if not atomic:
            async with self.engine.begin() as conn:
                await self._clear_index(conn)

        async for page in self._iter_pages(
            self._table.c[self.value_column], batch_size=batch_size
        ):
            rows = [{self.key: k, self.value_column: v} for k, v in page]
            await self._populate_system_columns(rows)
            total += len(rows)

            if atomic:
                buffered.extend(rows)
                continue

            async with self.engine.begin() as conn:
                await self._write_index(conn, rows)

        if atomic:
            async with self.engine.begin() as conn:
                await self._clear_index(conn)

                if buffered:
                    await self._write_index(conn, buffered)

        return total

    async def stream_rows(
        self, batch_size: int = 1000
    ) -> AsyncIterator[Sequence[tuple[K, V]]]:
        """Yield `(key, row)` pages of size `batch_size` of the rows in scope."""
        codec = self._codec
        payload_names = codec.columns
        payload_cols = [self._table.c[n] for n in payload_names]

        async for rows in self._iter_pages(
            *payload_cols, where=self._scoped(), batch_size=batch_size
        ):
            yield [
                (
                    row[0],
                    codec.decode(
                        dict(zip(payload_names, row[1:], strict=True)), row[0]
                    ),
                )
                for row in rows
            ]


__all__ = [
    "SQLAlchemy",
    "build_indexable_table",
    "reflect",
]
