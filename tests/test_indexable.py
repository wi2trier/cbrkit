"""Tests for indexable storage backends."""

import asyncio
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

import cbrkit
from cbrkit.errors import Conflict
from cbrkit.indexable import (
    Dense,
    Hybrid,
    LanceDBRetriever,
    Sparse,
    SQLiteVecRetriever,
    Unscoped,
    ZvecRetriever,
)
from cbrkit.typing import NumpyArray


class _engine_probe:
    """Observes an async engine: connections currently held, commits so far."""

    def __init__(self, engine: Any) -> None:
        from sqlalchemy import event

        self.held = 0
        self.commits = 0
        event.listen(engine.sync_engine, "checkout", self._checkout)
        event.listen(engine.sync_engine, "checkin", self._checkin)
        event.listen(engine.sync_engine, "commit", self._commit)

    def _checkout(self, *_: Any) -> None:
        self.held += 1

    def _checkin(self, *_: Any) -> None:
        self.held -= 1

    def _commit(self, *_: Any) -> None:
        self.commits += 1


async def _toy_embed(texts: Sequence[str]) -> Sequence[NumpyArray]:
    """Deterministic bag-of-keywords embedder."""
    import numpy as np

    vocab = ("red", "blue", "green", "car", "sky", "fruit")
    rows: list[NumpyArray] = []

    for text in texts:
        lowered = text.lower()
        vec = np.array([float(word in lowered) for word in vocab])

        if not vec.any():
            vec[:] = 1.0

        rows.append(vec)

    return rows


CASES = {"a": "red sedan car", "b": "blue sky", "c": "red apple fruit"}


def test_sqlite_vec_dense_sparse_hybrid(tmp_path: Path) -> None:
    """End-to-end dense / sparse / hybrid retrieval over a real SQLite file."""
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    from sqlalchemy.ext.asyncio import create_async_engine

    async def main() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cases.db")
        storage = cbrkit.indexable.SQLiteVec[str, str](
            engine,
            value_column="text",
            vector_dim=6,
            index_type="hybrid",
            embedder=_toy_embed,
            scope=Unscoped.TABLE,
        )
        assert await SQLiteVecRetriever(storage=storage)([({}, "red")]) == [({}, {})]
        await storage.create()
        await storage.create()
        await storage.put_index(CASES)
        assert await storage.has_index()
        assert await storage.get_index() == CASES

        dense = SQLiteVecRetriever(storage=storage, search=Dense(), limit=2)
        result = await cbrkit.cycle.run_query_async(None, "red car", dense)
        assert result.ranking[0] == "a"
        assert result.casebase["a"] == "red sedan car"

        sparse = SQLiteVecRetriever(storage=storage, search=Sparse())
        result = await cbrkit.cycle.run_query_async(None, "red", sparse)
        assert set(result.casebase) == {"a", "c"}

        hybrid = SQLiteVecRetriever(
            storage=storage,
            search=Hybrid(),
            limit=5,
            where=storage.sa_table.c.text.like("%fruit%"),
        )
        result = await cbrkit.cycle.run_query_async(None, "red car", hybrid)
        assert set(result.casebase) == {"c"}

        # deletes propagate to the vec0 shadow via trigger
        await storage.patch_index(delete=["a"])
        result = await cbrkit.cycle.run_query_async(None, "red car", dense)
        assert "a" not in result.casebase
        await engine.dispose()

    asyncio.run(main())


def test_sqlite_vec_reconcile(tmp_path: Path) -> None:
    """`reconcile` rebuilds once per fingerprint and recreates shadows of another declaration."""
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    from sqlalchemy.ext.asyncio import create_async_engine

    async def main() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cases.db")

        def storage(vector_column: str) -> cbrkit.indexable.SQLiteVec[str, str]:
            return cbrkit.indexable.SQLiteVec[str, str](
                engine,
                value_column="text",
                vector_column=vector_column,
                vector_dim=6,
                index_type="hybrid",
                embedder=_toy_embed,
                scope=Unscoped.TABLE,
            )

        old = storage("embedding")
        await old.create()
        await old.put_index(CASES)
        assert await old.reconcile("a")
        assert not await old.reconcile("a")
        assert await old.reconcile("b")

        new = storage("vector")
        assert await new.reconcile("b")
        assert not await new.reconcile("b")

        dense = SQLiteVecRetriever(storage=new, search=Dense(), limit=1)
        result = await cbrkit.cycle.run_query_async(None, "red car", dense)
        assert result.ranking == ("a",)

        sparse = SQLiteVecRetriever(storage=new, search=Sparse())
        result = await cbrkit.cycle.run_query_async(None, "red", sparse)
        assert set(result.casebase) == {"a", "c"}
        await engine.dispose()

    asyncio.run(main())


def test_sqlalchemy_populates_system_columns_outside_transactions() -> None:
    """Population runs outside the write transaction, except for `put_index`.

    Embedding inside a transaction keeps a pooled connection checked out
    (on PostgreSQL a session `idle in transaction`) for its whole duration.
    `put_index` is the deliberate exception: its rows are a diff against the
    current contents, so it builds them inside the transaction that read them.
    """
    pytest.importorskip("aiosqlite")

    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLAlchemy

    held_while_populating: list[int] = []

    async def main() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        probe = _engine_probe(engine)

        class probed(SQLAlchemy[str, str]):
            async def _populate_system_columns(
                self, rows: list[dict[str, Any]]
            ) -> None:
                held_while_populating.append(probe.held)

        storage = probed(engine, value_column="text", scope=Unscoped.TABLE)
        await storage.create()
        await storage.put_index({"a": "alpha"})
        await storage.patch_index(upsert={"b": "beta"})
        # Replaces the row matching the filter ("b"), leaving "a" untouched.
        await storage.replace_where(storage.sa_table.c.text == "beta", {"c": "gamma"})
        await storage.patch_index(upsert={"d": "delta"}, delete=["a"])
        assert await storage.get_index() == {"c": "gamma", "d": "delta"}
        await engine.dispose()

    asyncio.run(main())

    assert held_while_populating == [1, 0, 0, 0]


def test_sqlite_vec_host_table_reindex(tmp_path: Path) -> None:
    """Shadows are cbrkit-owned even when the data table is the host's.

    `create` creates the `vec0` / FTS5 shadows of a host table, and `reindex`
    backfills them from rows written outside cbrkit.  The host table carries a
    column named like the default `vector_column`: the shadow's columns are a
    separate namespace, so the two must coexist untouched.  Embedding stays
    outside the write transaction, since SQLite allows only a single writer.
    """
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLiteVec

    metadata = sa.MetaData()
    table = sa.Table(
        "cases",
        metadata,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("vector", sa.Text, server_default="host"),
    )

    async def main() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/host.db")
        probe = _engine_probe(engine)
        held_while_embedding: list[int] = []

        async def embed(texts: Sequence[str]) -> Sequence[NumpyArray]:
            held_while_embedding.append(probe.held)
            return await _toy_embed(texts)

        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert(),
                [{"id": "a", "text": "red sedan car"}, {"id": "b", "text": "blue sky"}],
            )

        storage = SQLiteVec[str, str](
            engine,
            table=table,
            value_column="text",
            vector_dim=6,
            index_type="hybrid",
            embedder=embed,
            scope=Unscoped.TABLE,
        )
        await storage.create()
        assert await storage.reindex(batch_size=1) == 2
        # Writes through cbrkit keep the shadows in sync from then on.
        await storage.patch_index(upsert={"c": "green apple"})
        await storage.patch_index(upsert={"c": "red apple fruit"})
        assert await storage.get_index() == {
            "a": "red sedan car",
            "b": "blue sky",
            "c": "red apple fruit",
        }

        async with engine.connect() as conn:
            assert set((await conn.scalars(sa.select(table.c.vector))).all()) == {
                "host"
            }
            vecs = await conn.scalar(sa.text('SELECT count(*) FROM "cases_vec"'))
            fts = await conn.scalar(sa.text('SELECT count(*) FROM "cases_fts"'))

        assert (vecs, fts) == (3, 3)
        # Two reindex pages plus the upserts, none of them holding a transaction.
        assert held_while_embedding == [0, 0, 0, 0]
        await engine.dispose()

    asyncio.run(main())


def test_pgvector_reindex_pages_outside_transactions(tmp_path: Path) -> None:
    """Re-embedding walks every page without ever holding a transaction open.

    `reindex` is dialect-agnostic, so a host table (a JSON column standing
    in for the vector) exercises it on SQLite.
    """
    pytest.importorskip("aiosqlite")
    pytest.importorskip("pgvector")

    import numpy as np
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import Pgvector

    class vec_json(sa.TypeDecorator[Any]):
        """JSON stand-in for `VECTOR`, which SQLite does not have."""

        impl = sa.JSON
        cache_ok = True

        def process_bind_param(self, value: Any, dialect: Any) -> Any:
            return None if value is None else [float(v) for v in value]

    metadata = sa.MetaData()
    table = sa.Table(
        "cases",
        metadata,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("vec", vec_json, nullable=True),
    )

    async def main() -> None:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cases.db")
        probe = _engine_probe(engine)
        held_while_embedding: list[int] = []
        batches: list[int] = []

        async def embed(texts: Sequence[str]) -> Sequence[NumpyArray]:
            held_while_embedding.append(probe.held)
            batches.append(len(texts))
            return [np.asarray([float(len(t))]) for t in texts]

        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert(),
                [{"id": f"k{i}", "text": "x" * i, "vec": None} for i in range(5)],
            )

        storage = Pgvector[str, str](
            engine,
            table=table,
            value_column="text",
            vector_column="vec",
            embedder=embed,
            scope=Unscoped.TABLE,
        )
        before = probe.commits
        assert await storage.reindex(batch_size=2) == 5
        paged_commits = probe.commits - before

        before = probe.commits
        await storage.reindex(batch_size=2, atomic=True)
        atomic_commits = probe.commits - before

        async with engine.connect() as conn:
            stored = dict(
                (await conn.execute(sa.select(table.c.id, table.c.vec))).all()
            )

        assert stored == {f"k{i}": [float(i)] for i in range(5)}
        # Every row reached, in pages of two, and nothing held while embedding.
        assert batches == [2, 2, 1] * 2
        assert held_while_embedding == [0] * 6
        # Paged commits once per page written, atomic buffers and commits once.
        assert atomic_commits == 1
        assert paged_commits > atomic_commits
        await engine.dispose()

    asyncio.run(main())


def test_pgvector_deletes_keys_through_one_array_parameter() -> None:
    """On PostgreSQL a key delete binds one array, whatever the key count."""
    pytest.importorskip("aiosqlite")
    pytest.importorskip("pgvector")

    from uuid import uuid4

    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import Pgvector

    storage = Pgvector[UUID, str](
        create_async_engine("sqlite+aiosqlite://"),
        key_type=sa.Uuid(),
        value_column="text",
        index_type="sparse",
        scope=Unscoped.TABLE,
    )
    keys = [uuid4() for _ in range(70_000)]
    stmt = sa.delete(storage.sa_table).where(*storage._keys_in(keys))
    compiled = stmt.compile(dialect=postgresql.dialect())

    assert "= ANY" in str(compiled)
    assert len(compiled.params) == 1



def test_pgvector_reconcile_replaces_columns_as_declared() -> None:
    """The columns that `reconcile` swaps are added as declared, an empty vector column nullable until filled."""
    pytest.importorskip("aiosqlite")
    pytest.importorskip("pgvector")

    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import Pgvector

    storage = Pgvector[str, str](
        create_async_engine("sqlite+aiosqlite://"),
        value_column="text",
        vector_dim=3,
        index_type="hybrid",
        tsvector_config=("german", "english"),
        embedder=_toy_embed,
        scope=Unscoped.TABLE,
    )
    dialect = postgresql.dialect()
    columns = storage.sa_table.c
    vector = storage._detached_column(columns.vector, nullable=True)
    comment = sa.schema.SetColumnComment(
        storage._detached_column(columns.vector, comment="m")
    )

    assert [
        *storage._replacement(vector, dialect),
        *storage._replacement(storage._detached_column(columns._tsvec), dialect),
        str(comment.compile(dialect=dialect)),
    ] == [
        "ALTER TABLE cases DROP COLUMN vector",
        "ALTER TABLE cases ADD COLUMN vector VECTOR(3)",
        "ALTER TABLE cases DROP COLUMN _tsvec",
        (
            "ALTER TABLE cases ADD COLUMN _tsvec TSVECTOR GENERATED ALWAYS AS "
            "(to_tsvector('german', text) || to_tsvector('english', text)) STORED NOT NULL"
        ),
        "COMMENT ON COLUMN cases.vector IS 'm'",
    ]

def test_sqlite_vec_uuid_keys_and_native_where() -> None:
    """Keys come back as the key column's Python type, here `UUID`.

    Covers the main table, both shadow tables (bound through the key type, so
    the joins match), and native SQLAlchemy `where` expressions.
    """
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    from uuid import uuid4

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLiteVec

    a, b, c = uuid4(), uuid4(), uuid4()
    cases = {a: "red sedan car", b: "blue sky", c: "red apple fruit"}

    async def main() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        storage = SQLiteVec[UUID, str](
            engine,
            key_type=sa.Uuid(),
            value_column="text",
            vector_dim=6,
            index_type="hybrid",
            embedder=_toy_embed,
            scope=Unscoped.TABLE,
        )
        await storage.create()
        text = storage.sa_table.c.text
        await storage.put_index(cases)
        assert await storage.get_index() == cases
        assert {k async for page in storage.stream_rows() for k, _ in page} == set(
            cases
        )
        assert await storage.keys_where(text == "blue sky") == [b]
        assert await storage.get_where(text == "blue sky") == {b: cases[b]}

        dense = SQLiteVecRetriever(storage=storage, search=Dense(), limit=1)
        _, sm = (await dense([({}, "red car")]))[0]
        assert list(sm) == [a]

        hybrid = SQLiteVecRetriever(storage=storage, limit=5, where=text.like("%red%"))
        _, sm = (await hybrid([({}, "red car")]))[0]
        assert set(sm) == {a, c}

        assert await storage.delete_where(text.like("%fruit%")) == [c]
        await storage.patch_index(delete=[b])
        _, sm = (await hybrid([({}, "red")]))[0]
        assert set(sm) == {a}
        await engine.dispose()

    asyncio.run(main())


@pytest.mark.parametrize("key_type", ["integer", "text"])
def test_sqlalchemy_allocates_keys_in_the_write_transaction(key_type: str) -> None:
    pytest.importorskip("aiosqlite")

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLAlchemy

    integer = key_type == "integer"

    async def main() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        storage = SQLAlchemy[Any, str](
            engine,
            value_column="value",
            key_type=sa.Integer() if integer else sa.Text(),
            scope=Unscoped.TABLE,
        )
        await storage.create()
        await storage.patch_index(upsert={4: "a"} if integer else {"a": "a"})
        keys = await storage.patch_index(
            delete=[4] if integer else ["a"], add=["b", "c"]
        )

        assert dict(await storage.get_index()) == dict(
            zip(keys, ["b", "c"], strict=True)
        )
        assert keys == [5, 6] if integer else all(len(k) == 36 for k in keys)

        inserted = 9 if integer else "z"
        added = await storage.patch_index(add=["e"], insert={inserted: "d"})

        assert added == [10] if integer else len(added[0]) == 36

        with pytest.raises(Conflict):
            await storage.patch_index(insert={inserted: "f"})

        await engine.dispose()

    asyncio.run(main())


def test_sqlalchemy_scope_restricts_reads_and_writes() -> None:
    """Storages scoped to one owner share a host table of plain text without seeing or touching the rows of another."""
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.errors import Refused
    from cbrkit.indexable import SQLiteVec

    def owner_of(ctx: Any) -> str:
        return ctx.get_current_parameters()["key"].split(":")[0]

    table = sa.Table(
        "cases",
        sa.MetaData(),
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("owner", sa.Text, default=owner_of),
        sa.Column("text", sa.Text),
    )

    def scoped(owner: str) -> SQLiteVec[str, str]:
        return SQLiteVec[str, str](
            engine,
            table=table,
            value_column="text",
            index_type="sparse",
            scope=table.c.owner == owner,
        )

    def rows(owner: str, *texts: str) -> dict[str, str]:
        return {f"{owner}:{i}": t for i, t in enumerate(texts)}

    engine = create_async_engine("sqlite+aiosqlite://")

    async def main() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(table.create)

        alice, bob = scoped("alice"), scoped("bob")
        await alice.create()
        await alice.put_index(rows("alice", "red car", "blue sky"))
        await bob.put_index(rows("bob", "red car"))
        await alice.put_index(rows("alice", "red car"))
        assert await alice.get_index() == rows("alice", "red car")
        assert await bob.get_index() == rows("bob", "red car")

        await alice.patch_index(
            upsert=rows("alice", "green car"), expected=rows("alice", "red car")
        )

        for write in (
            alice.put_index(rows("bob", "stolen")),
            alice.patch_index(upsert={"bob:1": "added"}),
            alice.replace_where(sa.true(), rows("bob", "stolen")),
        ):
            with pytest.raises(Refused):
                await write

        assert await alice.get_index() == rows("alice", "green car")
        assert await bob.get_index() == rows("bob", "red car")

        retriever = SQLiteVecRetriever(storage=alice)
        _, sm = (await retriever([({}, "car")]))[0]
        assert set(sm) == {"alice:0"}

        assert await alice.narrow(table.c.owner == "bob").get_index() == {}
        assert await alice.narrow(sa.true()).get_index() == rows("alice", "green car")

        with pytest.raises(Refused):
            await alice.reindex()

        await engine.dispose()

    asyncio.run(main())


def test_sqlalchemy_upsert_keeps_unwritten_columns_and_applies_onupdate() -> None:
    """A replaced row keeps its server default and gets the `onupdate`, evaluated once per statement."""
    pytest.importorskip("aiosqlite")

    import itertools

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLAlchemy

    counter = itertools.count(1)
    table = sa.Table(
        "cases",
        sa.MetaData(),
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("text", sa.Text),
        sa.Column("created_at", sa.Integer, server_default=sa.text("(random())")),
        sa.Column("updated_at", sa.Integer, default=0, onupdate=lambda: next(counter)),
    )
    engine = create_async_engine("sqlite+aiosqlite://")
    storage = SQLAlchemy[str, str](
        engine, table=table, value_column="text", scope=Unscoped.TABLE
    )

    async def stamps() -> list[tuple[int, int]]:
        async with engine.connect() as conn:
            stmt = sa.select(table.c.created_at, table.c.updated_at).order_by(
                table.c.key
            )
            return [(c, u) for c, u in (await conn.execute(stmt)).all()]

    async def main() -> None:
        async with engine.begin() as conn:
            await conn.run_sync(table.create)

        await storage.put_index({"a": "alpha", "b": "beta"})
        before = await stamps()
        await storage.put_index({"a": "gamma", "b": "delta"})

        assert [u for _, u in before] == [0, 0]
        # each upsert statement evaluates the onupdate once, even when nothing conflicts
        assert await stamps() == [(c, 2) for c, _ in before]
        assert await storage.get_index() == {"a": "gamma", "b": "delta"}
        await engine.dispose()

    asyncio.run(main())


def test_sqlalchemy_rejects_an_onupdate_reading_the_context() -> None:
    pytest.importorskip("aiosqlite")

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLAlchemy

    def text_length(ctx: Any) -> int:
        return len(ctx.get_current_parameters()["text"])

    table = sa.Table(
        "cases",
        sa.MetaData(),
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("text", sa.Text),
        sa.Column("updated_at", sa.Integer, onupdate=text_length),
    )

    with pytest.raises(TypeError, match="updated_at"):
        SQLAlchemy[str, str](
            create_async_engine("sqlite+aiosqlite://"),
            table=table,
            value_column="text",
            scope=Unscoped.TABLE,
        )


def test_sqlalchemy_reads_a_reflected_table() -> None:
    pytest.importorskip("aiosqlite")

    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import SQLAlchemy, reflect

    async def main() -> None:
        engine = create_async_engine("sqlite+aiosqlite://")
        owned = SQLAlchemy[str, str](
            engine, key_column="id", value_column="text", scope=Unscoped.TABLE
        )
        await owned.create()
        await owned.put_index({"a": "alpha"})
        storage = SQLAlchemy[str, Any](
            engine, table=await reflect(engine, "cases"), scope=Unscoped.TABLE
        )

        assert storage.key == "id"
        assert await storage.get_index() == {"a": {"text": "alpha"}}
        await engine.dispose()

    asyncio.run(main())


def test_lancedb_put_and_predicate_helpers(tmp_path: Path) -> None:
    """Exercise the diff of put_index, native predicate helpers, and key escaping."""
    lancedb = pytest.importorskip("lancedb")

    @dataclasses.dataclass
    class Doc:
        value: str
        source: str

    def _docs(cases: dict[str, str]) -> dict[str, Doc]:
        return {k: Doc(v, k.split("::", maxsplit=1)[0]) for k, v in cases.items()}

    async def main() -> None:
        storage = cbrkit.indexable.LanceDB[str, Doc](
            await lancedb.connect_async(tmp_path),
            index_type="sparse",
            model=Doc,
        )
        assert not await storage.has_index()
        assert await LanceDBRetriever(storage=storage)([({}, "x")]) == [({}, {})]
        await storage.put_index(
            _docs({"doc-a::0": "alpha", "doc-a::1": "beta", "quote's::0": "gamma"})
        )
        await storage.put_index(
            _docs(
                {"doc-a::0": "alpha updated", "doc-a::1": "beta", "doc-b::0": "delta"}
            )
        )

        assert await storage.get_index() == _docs(
            {"doc-a::0": "alpha updated", "doc-a::1": "beta", "doc-b::0": "delta"}
        )
        doc_a = "source = 'doc-a'"
        assert set(await storage.keys_where(doc_a)) == {"doc-a::0", "doc-a::1"}
        assert set(await storage.get_where(doc_a)) == {"doc-a::0", "doc-a::1"}
        assert set(await storage.delete_where(doc_a)) == {"doc-a::0", "doc-a::1"}
        assert await storage.get_index() == _docs({"doc-b::0": "delta"})

        _, sm = (await LanceDBRetriever(storage=storage)([({}, "delta")]))[0]
        assert list(sm) == ["doc-b::0"]

    asyncio.run(main())


def test_zvec_dense_fts_hybrid(tmp_path: Path) -> None:
    """End-to-end dense / FTS-sparse / hybrid retrieval over a real zvec collection."""
    pytest.importorskip("zvec")

    def _build() -> cbrkit.indexable.Zvec[str, str]:
        return cbrkit.indexable.Zvec[str, str](
            tmp_path / "cases",
            index_type="hybrid",
            embedder=_toy_embed,
        )

    async def main() -> None:
        storage = _build()
        assert not await storage.has_index()
        assert await ZvecRetriever(storage=storage)([({}, "red")]) == [({}, {})]
        await storage.put_index(CASES)
        assert await storage.get_index() == CASES

        dense = ZvecRetriever(storage=storage, search=Dense(), limit=2)
        cb, sm = (await dense([({}, "red car")]))[0]
        assert next(iter(sm)) == "a"
        assert cb["a"] == "red sedan car"

        sparse = ZvecRetriever(storage=storage, search=Sparse())
        _, sm = (await sparse([({}, "red")]))[0]
        assert set(sm) == {"a", "c"}

        hybrid = ZvecRetriever(storage=storage, search=Hybrid(), limit=5)
        _, sm = (await hybrid([({}, "red car")]))[0]
        assert {"a", "c"} <= set(sm)

        # the fusion honours the weights, so a dense-only weighting keeps the dense ranking
        _, dense_sm = (
            await ZvecRetriever(storage=storage, search=Dense())([({}, "red car")])
        )[0]
        weighted = ZvecRetriever(storage=storage, search=Hybrid(weights=(1.0, 0.0)))
        _, sm = (await weighted([({}, "red car")]))[0]
        assert list(sm) == list(dense_sm)

        # deletes propagate to the FTS index and persist for a new storage
        await storage.patch_index(delete=["a"])
        retriever = ZvecRetriever(storage=_build(), search=Sparse())
        _, sm = (await retriever([({}, "red")]))[0]
        assert set(sm) == {"c"}

        # a keyed insert conflicts with a stored case
        await storage.patch_index(insert={"a": "red apple"})
        with pytest.raises(Conflict):
            await storage.patch_index(insert={"c": "green"})

    asyncio.run(main())


def test_codec_restores_the_key_into_a_field_the_model_generates() -> None:
    from cbrkit.indexable._common import make_codec

    @dataclasses.dataclass
    class Row:
        text: str
        id: str = dataclasses.field(init=False, default="generated")

    codec = make_codec(Row, "value", key_column="id")

    assert codec.decode({"text": "a"}, "k").id == "k"


def test_dot_metric_ranks_the_largest_product_first(tmp_path: Path) -> None:
    """The dense scores of LanceDB and zvec grow with the dot product."""
    lancedb = pytest.importorskip("lancedb")
    pytest.importorskip("zvec")

    from cbrkit.sim.embed.metrics import Dot

    async def main() -> None:
        lance = cbrkit.indexable.LanceDB[str, str](
            await lancedb.connect_async(tmp_path / "lance"),
            embedder=_toy_embed,
            metric=Dot(),
        )
        zvec = cbrkit.indexable.Zvec[str, str](
            tmp_path / "zvec", embedder=_toy_embed, metric=Dot()
        )

        for storage, retriever in (
            (lance, LanceDBRetriever(storage=lance)),
            (zvec, ZvecRetriever(storage=zvec)),
        ):
            await storage.put_index(CASES)
            _, sm = (await retriever([({}, "red car")]))[0]
            assert list(sm)[:2] == ["a", "c"]

    asyncio.run(main())
