"""PostgreSQL/pgvector storage backend."""

from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, ClassVar, Literal, override

import numpy as np
import sqlalchemy as sa
from pgvector.sqlalchemy import HALFVEC
from pgvector.sqlalchemy import Vector as PGVECTOR
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncConnection

from .._util import identity
from ..errors import Refused
from ..helpers import dist2sim
from ..sim.embed.metrics import Cosine, Dot, Euclidean
from ..typing import EmbeddingFunc, NumpyArray
from ._common import VectorMetric
from ._retriever import (
    IndexType,
    check_embedder,
    dense_embedder,
    embed_rows,
    has_dense,
    has_sparse,
)
from ._sqlalchemy import SQLAlchemy
from ._sqlalchemy_retriever import SQLAlchemyVectorRetriever, check_vectors

# vector_type -> (HNSW opclass prefix, SQLAlchemy column type).  Both
# float32 (`vector`) and half-precision (`halfvec`, ~2x smaller) share the
# same distance comparator methods, so only the column type and opclass
# prefix differ.
_PG_VECTOR_TYPES: Mapping[Literal["float32", "halfvec"], tuple[str, Any]] = {
    "float32": ("vector", PGVECTOR),
    "halfvec": ("halfvec", HALFVEC),
}


@dataclass(slots=True, frozen=True)
class _PGMetric:
    """Per-metric pgvector configuration."""

    ops_suffix: str
    """Operator-class suffix, the full HNSW opclass is
    `f"{type_prefix}_{ops_suffix}_ops"` (e.g. `vector_cosine_ops`,
    `halfvec_cosine_ops`)."""
    distance_attr: str
    """`pgvector.sqlalchemy` comparator method returning the distance expression."""
    sim_from_distance: Callable[[float], float]
    """Convert a raw pgvector distance to a similarity score."""


_PG_METRICS: Mapping[type[VectorMetric], _PGMetric] = {
    Cosine: _PGMetric("cosine", "cosine_distance", dist2sim),
    Dot: _PGMetric("ip", "max_inner_product", lambda d: -d),
    Euclidean: _PGMetric("l2", "l2_distance", dist2sim),
}


def _fts_configs(config: Sequence[str]) -> tuple[str, ...]:
    """Check the PostgreSQL FTS configurations, at least one.

    A single configuration is the common case.
    Several enable true multi-language indexing: the stored
    `tsvector` and the query `tsquery` become the union of one
    `to_tsvector` / `plainto_tsquery` per configuration, so a corpus
    mixing e.g. German and English is stemmed correctly for both.
    """
    configs = tuple(config)

    if not configs:
        raise ValueError("tsvector_config sequence must be non-empty")

    return configs


def _fts_combine(terms: Iterable[Any]) -> Any:
    """Fold per-config FTS expressions with the PostgreSQL `||` operator.

    `||` concatenates `tsvector` operands and ORs `tsquery` operands,
    so this combines one `to_tsvector` / `plainto_tsquery` per configured
    language into a single multilingual expression.
    """
    return reduce(lambda acc, term: acc.op("||")(term), terms)


_GENERATED_EXPR = sa.text(
    "SELECT pg_get_expr(d.adbin, d.adrelid) "
    "FROM pg_attrdef d JOIN pg_attribute a "
    "ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
    "WHERE d.adrelid = CAST(:table AS regclass) AND a.attname = :column"
)
"""The deparsed expression of a generated column."""


def tsvector_computed(source: str, config: Sequence[str] = ("simple",)) -> sa.Computed:
    """Persisted `to_tsvector` clause over `source`, for host model columns.

    Use this in a host-declared mapped model's `TSVECTOR` column and pass
    the same `config` to the storage's `tsvector_config` so the
    generated column and the query `tsquery` stay in lockstep, declaring
    the clause by hand in two places that must agree is the usual source of
    silently broken sparse search.

    A single configuration builds `to_tsvector(config, source)`, several
    build the union `to_tsvector(c1, source) || to_tsvector(c2, source) || ...`
    for true multi-language indexing (e.g. `("german", "english")`),
    matching the OR-combined `tsquery` the retriever
    builds from the same list.  The default `"simple"` is
    language-agnostic.
    """
    expr = _fts_combine(
        sa.func.to_tsvector(sa.literal(cfg), sa.column(source))
        for cfg in _fts_configs(config)
    )

    return sa.Computed(expr, persisted=True)


@dataclass(slots=True, frozen=True, kw_only=True)
class Pgvector[K: Hashable, V = Mapping[str, Any]](SQLAlchemy[K, V]):
    """PostgreSQL/pgvector tabular storage.

    Extends `SQLAlchemy` with two system columns derived from
    the inherited `value_column` (the embeddable text column):

    - `vector_column` carries a dense embedding (computed in Python via
      `embedder` and inserted on write).
    - `tsvector_column` is a Postgres-generated `TSVECTOR` over the
      value column.

    The `value_column` is the single text-field knob shared with the other
    backends: with no `model` / `table` cbrkit builds a
    one-column table and `V = str`, with a `model` / host table it names
    which existing column is embedded and FTS-indexed.
    Writes use `INSERT ... ON CONFLICT DO UPDATE` in a single round trip.

    The simplest setup stores plain strings (`V = str`) in a cbrkit-built
    table::

        storage = Pgvector(
            engine,
            value_column="body",
            vector_dim=384,
            embedder=embed,
            scope=Unscoped.TABLE,
        )
        await storage.create()
        await storage.put_index({"a": "red sedan"})

    For typed rows, pass a SQLAlchemy mapped `model` (host-created) that
    declares the re-exported `PGVECTOR` and `TSVECTOR` columns
    and point `value_column` at the embeddable column::

        from cbrkit.indexable import PGVECTOR, TSVECTOR, tsvector_computed
        from sqlalchemy.orm import Mapped, mapped_column

        class Chunk(Base):
            __tablename__ = "chunks"
            id: Mapped[str] = mapped_column(primary_key=True)
            body: Mapped[str]
            vector: Mapped[Any] = mapped_column(PGVECTOR(384), nullable=False)
            _tsvec: Mapped[Any] = mapped_column(
                TSVECTOR(), tsvector_computed("body"), nullable=False
            )

        storage = Pgvector(engine, model=Chunk, value_column="body", scope=..., ...)
        await storage.create()

    Pass the same `tsvector_config` to `tsvector_computed` and the
    storage so the generated column and the query side agree.
    `reconcile` keeps a host table in line with them before the storage serves:
    it fits the live index columns to their declaration and rebuilds the vectors once what made them changed.

    Args:
        vector_column: Column of the embeddings.
        tsvector_column: Column of the generated `TSVECTOR`, underscored to mark it as owned by cbrkit.
        vector_dim: Embedding dimension, required for `manage_schema=True`
            and an `index_type` of `"dense"` or `"hybrid"`.
        vector_type: Stored element type, `"float32"` (`vector`, exact)
            or `"halfvec"` (half precision, ~2x smaller with negligible
            recall loss).  Both share the same distance metrics, for a
            host-supplied `model`, declare the column with the matching
            re-exported type (`PGVECTOR` / `HALFVEC`).
        index_type: `"dense"` (HNSW only), `"sparse"` (GIN only), or
            `"hybrid"` (both).
        tsvector_config: PostgreSQL FTS configurations for the
            `tsvector_column` generator and the matching query
            `tsquery`.  Defaults to the language-agnostic `"simple"`
            (no stemming or stop-words) so non-English text is never
            silently mis-stemmed.  Pass one configuration (e.g.
            `("english",)`) for a known monolingual corpus, or several
            (e.g. `("german", "english")`) for true multi-language
            indexing.
        metric: Distance metric used by the HNSW index (and mirrored
            at search time by the retriever), `Cosine`, `Dot`, or `Euclidean`.
        embedder: Embedding function.  Required for `"dense"` /
            `"hybrid"` index types.
    """

    _PARAM_LIMIT: ClassVar[int] = 65_535

    vector_column: str = "vector"
    tsvector_column: str = "_tsvec"
    vector_dim: int | None = None
    vector_type: Literal["float32", "halfvec"] = "float32"
    index_type: IndexType = "dense"
    tsvector_config: tuple[str, ...] = ("simple",)
    metric: VectorMetric = field(default_factory=Cosine)
    embedder: EmbeddingFunc[str, NumpyArray] | None = None

    def _validate_init(self) -> None:
        super(Pgvector, self)._validate_init()
        check_vectors("pgvector", self.value_column, self.vector_dim)
        check_embedder(self.index_type, self.embedder)

        if (
            self.manage_schema
            and self._builds_own_schema
            and has_dense(self.index_type)
            and self.vector_dim is None
        ):
            raise ValueError(
                "vector_dim is required when manage_schema=True and "
                f"index_type={self.index_type!r}"
            )

        if has_sparse(self.index_type):
            _fts_configs(self.tsvector_config)

    def _system_columns(self) -> Sequence[sa.Column[Any]]:
        cols: list[sa.Column[Any]] = []
        if has_dense(self.index_type):
            assert self.vector_dim is not None
            _, col_type = _PG_VECTOR_TYPES[self.vector_type]
            cols.append(
                sa.Column(self.vector_column, col_type(self.vector_dim), nullable=False)
            )
        if has_sparse(self.index_type):
            assert self.value_column is not None
            cols.append(
                sa.Column(
                    self.tsvector_column,
                    TSVECTOR(),
                    tsvector_computed(self.value_column, self.tsvector_config),
                    nullable=False,
                )
            )
        return cols

    def _system_column_names(self) -> frozenset[str]:
        names: set[str] = set()
        if has_dense(self.index_type):
            names.add(self.vector_column)
        if has_sparse(self.index_type):
            names.add(self.tsvector_column)
        return frozenset(names)

    def _pre_create_ddl(self, sync_conn: sa.Connection) -> None:
        if has_dense(self.index_type):
            sync_conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    def _system_indexes(self, column: str) -> list[sa.Index]:
        """The indexes of a system column, those the host declares or else the one cbrkit names."""
        target = self._table.c[column]

        if declared := [
            index
            for index in self._table.indexes
            if index.columns.contains_column(target)
        ]:
            return declared

        if column == self.vector_column:
            prefix, _ = _PG_VECTOR_TYPES[self.vector_type]
            opclass = f"{prefix}_{_PG_METRICS[type(self.metric)].ops_suffix}_ops"

            return [
                sa.Index(
                    f"ix_{self._table.name}_{column}",
                    target,
                    postgresql_using="hnsw",
                    postgresql_ops={column: opclass},
                )
            ]

        return [
            sa.Index(f"ix_{self._table.name}_{column}", target, postgresql_using="gin")
        ]

    def _create_system_indexes(self, sync_conn: sa.Connection) -> None:
        for column in self._system_column_names():
            for index in self._system_indexes(column):
                index.create(sync_conn, checkfirst=True)

    async def _populate_system_columns(self, rows: list[dict[str, Any]]) -> None:
        """Embed the text column into the vector column of each row.

        The vector stays an `numpy.ndarray` rather than being flattened
        to `list[float]`: `VECTOR` and `HALFVEC` both bind an
        ndarray to the identical literal, and a page of them is an order of
        magnitude smaller in memory (a 384-dim row is ~1.6 kB as float32
        against ~12 kB as boxed Python floats).  That matters wherever a page
        is held rather than written straight through, see
        `SQLAlchemy.reindex`.
        """
        assert self.value_column is not None

        if (embedder := dense_embedder(self.index_type, self.embedder)) is not None:
            await embed_rows(
                embedder, rows, self.value_column, self.vector_column, np.asarray
            )

    def _keys_in(self, keys: Sequence[K]) -> Iterator[sa.ColumnElement[bool]]:
        """Match `keys` through `= ANY(:keys)`, one array parameter for any count."""
        if not keys:
            return

        kc = self._table.c[self.key]
        yield kc == sa.any_(sa.bindparam("keys", list(keys), type_=ARRAY(kc.type)))

    @property
    def _reindexed(self) -> bool:
        """Whether `reindex` re-embeds the rows, the generated `TSVECTOR` needs no rebuild."""
        return has_dense(self.index_type)

    async def _write_index(
        self, conn: AsyncConnection, rows: list[dict[str, Any]]
    ) -> None:
        """UPDATE the vector column of every row whose text is still unchanged.

        Matching on the text as well as the key makes this a compare-and-set: a
        row edited mid-page keeps the vector its editing writer stored,
        so `reindex` may walk more rows than it updates.
        """
        assert self.value_column is not None
        kc = self._table.c[self.key]
        tc = self._table.c[self.value_column]
        await conn.execute(
            sa.update(self._table)
            .where(kc == sa.bindparam("k"), tc == sa.bindparam("t"))
            .values({self.vector_column: sa.bindparam("v")}),
            [
                {
                    "k": row[self.key],
                    "t": row[self.value_column],
                    "v": row[self.vector_column],
                }
                for row in rows
            ],
        )

    async def reconcile(self, fingerprint: str, /) -> bool:
        """Fit the index columns of the table to their declaration and rebuild the vectors if what made them changed.

        A vector column of another type than declared, such as of another dimension, is swapped for an empty nullable one,
        since vectors cannot be cast across dimensions,
        and a generated `TSVECTOR` column of another expression, such as of other configurations,
        is created again, which PostgreSQL fills from the text.
        The vectors are rebuilt if their column was swapped or carries another fingerprint as its comment,
        and the fingerprint is written only after the rebuild, so a crash at any point makes the next call rebuild again.
        The rebuild commits page by page, see `reindex`, so call it before the storage serves.

        Args:
            fingerprint: What determines the stored vectors, such as the embedding model and its document prefix,
                but not the code that computes them, after a change of which the host calls `reindex` itself.

        Returns:
            Whether the vectors were rebuilt.

        Raises:
            Refused: When the storage is scoped, since a rebuild writes every row.
        """
        self._require_unscoped()

        async with self.engine.begin() as conn:
            if has_sparse(self.index_type):
                await conn.run_sync(self._fit_tsvector)

            if not has_dense(self.index_type):
                return False

            if await conn.run_sync(self._fit_vector, fingerprint):
                return False

            for index in self._system_indexes(self.vector_column):
                await conn.run_sync(index.drop, checkfirst=True)

        _ = await self.reindex()

        async with self.engine.begin() as conn:
            await conn.run_sync(self._complete_vector, fingerprint)

        return True

    def _alteration(self, dialect: sa.Dialect, change: str) -> str:
        return f"ALTER TABLE {dialect.identifier_preparer.format_table(self._table)} {change}"

    def _alter(self, conn: sa.Connection, change: str) -> None:
        conn.execute(sa.text(self._alteration(conn.dialect, change)))

    def _replace_column(self, conn: sa.Connection, column: sa.Column[Any]) -> None:
        """Drop a column with its indexes and add it as declared, empty or generated."""
        for statement in self._replacement(column, conn.dialect):
            conn.execute(sa.text(statement))

    def _replacement(
        self, column: sa.Column[Any], dialect: sa.Dialect
    ) -> tuple[str, str]:
        """The statements that drop a column of the table and add it again, see `_replace_column`."""
        dropped = dialect.identifier_preparer.format_column(column)
        added = sa.schema.CreateColumn(column).compile(dialect=dialect)

        return (
            self._alteration(dialect, f"DROP COLUMN {dropped}"),
            self._alteration(dialect, f"ADD COLUMN {added}"),
        )

    def _fit_vector(self, conn: sa.Connection, fingerprint: str) -> bool:
        """Swap a vector column of another type for an empty nullable one, returning whether it holds the vectors of the fingerprint."""
        declared = self._table.c[self.vector_column]
        live = {
            column["name"]: column
            for column in sa.inspect(conn).get_columns(
                self._table.name, schema=self._table.schema
            )
        }.get(declared.name)

        if live is None:
            raise Refused(
                f"Table {self._table.fullname!r} has no vector column {declared.name!r}. "
                "Use the existing column name or migrate the table."
            )

        if live["type"].compile(conn.dialect) == declared.type.compile(conn.dialect):
            return not live["nullable"] and live.get("comment") == fingerprint

        self._replace_column(conn, self._detached_column(declared, nullable=True))

        return False

    def _complete_vector(self, conn: sa.Connection, fingerprint: str) -> None:
        """Restore `NOT NULL`, build the indexes once over the filled column, and record the fingerprint."""
        column = self._detached_column(
            self._table.c[self.vector_column], comment=fingerprint
        )
        preparer = conn.dialect.identifier_preparer
        self._alter(conn, f"ALTER COLUMN {preparer.format_column(column)} SET NOT NULL")
        conn.execute(sa.schema.SetColumnComment(column))

        for index in self._system_indexes(self.vector_column):
            index.create(conn, checkfirst=True)

    def _fit_tsvector(self, conn: sa.Connection) -> None:
        """Create the generated `TSVECTOR` column again if its live expression differs from the declared one.

        The declared expression is created on a temporary table that the transaction drops,
        so PostgreSQL deparses both alike and they compare as equal strings.
        """
        assert self.value_column is not None
        declared = self._table.c[self.tsvector_column]
        computed = declared.computed
        assert computed is not None
        probe = sa.Table(
            "_cbrkit_tsvector_probe",
            sa.MetaData(),
            sa.Column(self.value_column, self._table.c[self.value_column].type),
            sa.Column(
                declared.name,
                declared.type,
                sa.Computed(computed.sqltext, persisted=True),
            ),
            schema="pg_temp",
            prefixes=["TEMPORARY"],
            postgresql_on_commit="DROP",
        )
        probe.create(conn)
        preparer = conn.dialect.identifier_preparer
        live, expected = (
            conn.execute(
                _GENERATED_EXPR,
                {"table": table, "column": declared.name},
            ).scalar_one_or_none()
            for table in (
                preparer.format_table(self._table),
                preparer.format_table(probe),
            )
        )

        if live == expected:
            return

        self._replace_column(conn, self._detached_column(declared))

        for index in self._system_indexes(self.tsvector_column):
            index.create(conn, checkfirst=True)

    def _detached_column(
        self,
        column: sa.Column[Any],
        *,
        nullable: bool | None = None,
        comment: str | None = None,
    ) -> sa.Column[Any]:
        """A copy of a column of the table, on a table of the same name, with its nullability or comment changed."""
        computed = column.computed
        copy = sa.Column(
            column.name,
            column.type,
            *(
                ()
                if computed is None
                else (sa.Computed(computed.sqltext, persisted=True),)
            ),
            nullable=column.nullable if nullable is None else nullable,
            comment=column.comment if comment is None else comment,
        )
        _ = sa.Table(self._table.name, sa.MetaData(), copy, schema=self._table.schema)

        return copy


@dataclass(slots=True, frozen=True, kw_only=True)
class PgvectorRetriever[K: Hashable](SQLAlchemyVectorRetriever[K, Pgvector[K, Any]]):
    """Search of the `value_column` of a `Pgvector` storage.

    Several retrievers can share one storage to search it with different settings.
    The casebase of a result holds the text column, read full rows from the storage.
    The queries of one call run sequentially on a single connection.
    """

    @override
    def _dense_stmt(self, qvec: NumpyArray, k: int | None) -> sa.Select[Any, Any, Any]:
        storage = self.storage
        assert storage.value_column is not None
        table = storage.sa_table
        metric = _PG_METRICS[type(storage.metric)]
        distance = getattr(table.c[storage.vector_column], metric.distance_attr)(
            np.asarray(qvec).tolist()
        )

        return sa.select(
            table.c[storage.key],
            table.c[storage.value_column],
            distance.label("distance"),
        ).order_by(distance)

    @override
    def _sparse_stmt(self, query: str) -> sa.Select[Any, Any, Any]:
        storage = self.storage
        assert storage.value_column is not None
        table = storage.sa_table
        tsv = table.c[storage.tsvector_column]
        tsq = _fts_combine(
            sa.func.plainto_tsquery(cfg, query) for cfg in storage.tsvector_config
        )
        rank = sa.func.ts_rank(tsv, tsq)

        return (
            sa.select(
                table.c[storage.key],
                table.c[storage.value_column],
                rank.label("score"),
            )
            .where(tsv.op("@@")(tsq))
            .order_by(rank.desc())
        )

    @property
    @override
    def _dense_score(self) -> Callable[[float], float]:
        return _PG_METRICS[type(self.storage.metric)].sim_from_distance

    @property
    @override
    def _sparse_score(self) -> Callable[[float], float]:
        return identity


__all__ = [
    "HALFVEC",
    "PGVECTOR",
    "TSVECTOR",
    "Pgvector",
    "PgvectorRetriever",
    "tsvector_computed",
]
