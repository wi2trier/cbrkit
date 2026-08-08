"""Tests for indexable storage backends."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import cbrkit
from cbrkit.typing import NumpyArray


class _engine_probe:
    """Observes an async engine: connections currently held, commits so far.

    Constructed before the engine exists so tests can close over it, then
    :meth:`attach` once they have one.
    """

    def __init__(self) -> None:
        self.held = 0
        self.commits = 0

    def attach(self, engine: Any) -> None:
        from sqlalchemy import event

        event.listen(engine.sync_engine, "checkout", self._checkout)
        event.listen(engine.sync_engine, "checkin", self._checkin)
        event.listen(engine.sync_engine, "commit", self._commit)

    def _checkout(self, *_: Any) -> None:
        self.held += 1

    def _checkin(self, *_: Any) -> None:
        self.held -= 1

    def _commit(self, *_: Any) -> None:
        self.commits += 1


def _toy_embed(texts: Sequence[str]) -> Sequence[NumpyArray]:
    """Deterministic bag-of-keywords embedder for sqlite_vec tests."""
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


def test_sqlite_vec_dense_sparse_hybrid(tmp_path: Path) -> None:
    """End-to-end dense / sparse / hybrid retrieval over a real SQLite file."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    from cbrkit.filter import Like

    url = f"sqlite+aiosqlite:///{tmp_path}/cases.db"
    cases = {"a": "red sedan car", "b": "blue sky", "c": "red apple fruit"}

    storage = cbrkit.indexable.sqlite_vec[str, str](
        url=url,
        value_column="text",
        vector_dim=6,
        index_type="hybrid",
        conversion_func=_toy_embed,
    )
    storage.put_index(cases)
    assert storage.has_index()
    assert storage.index == cases

    # dense: "red car" is closest to "a" (red + car)
    dense = cbrkit.retrieval.indexable.sqlite_vec(storage, search_type="dense", limit=2)
    cb, sm = dense([({}, "red car")])[0]
    assert next(iter(sm)) == "a"
    assert cb["a"] == "red sedan car"

    # sparse: FTS5 keyword "red" hits "a" and "c", not "b"
    sparse = cbrkit.retrieval.indexable.sqlite_vec(storage, search_type="sparse")
    _, sm = sparse([({}, "red")])[0]
    assert set(sm) == {"a", "c"}

    # hybrid + filter: restrict to fruit-ish rows via a LIKE WHERE clause
    hybrid = cbrkit.retrieval.indexable.sqlite_vec(
        storage,
        search_type="hybrid",
        limit=5,
        where=Like(column="text", pattern="%fruit%"),
    )
    _, sm = hybrid([({}, "red car")])[0]
    assert set(sm) == {"c"}

    # deletes propagate to the vec0 shadow via trigger
    storage.delete_index(["a"])
    cb, _ = dense([({}, "red car")])[0]
    assert "a" not in cb

    storage.close()


def test_sqlalchemy_async_populates_system_columns_off_loop(tmp_path: Path) -> None:
    """Population runs in a worker thread and outside the write transaction.

    Regressions: pgvector's ``conversion_func`` ran on the event loop, freezing
    the host application for the duration of an embedding batch, and it ran
    inside the write transaction, keeping a pooled connection checked out (on
    PostgreSQL a session ``idle in transaction``) for just as long.
    ``put_index`` is the deliberate exception: its rows are a diff against the
    current contents, so it must build them inside the transaction that read
    them.  Caller-owned *data* must still be read on the loop thread, since
    mappings and ORM/model values may be thread-affine.
    """
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")

    import asyncio
    import threading
    from collections.abc import Iterator, Mapping

    from cbrkit.filter import Eq
    from cbrkit.indexable import sqlalchemy_async

    populate_idents: list[int] = []
    read_idents: list[int] = []
    held_while_populating: list[int] = []
    engine_probe = _engine_probe()

    class loop_affine(Mapping[str, str]):
        """Caller-owned mapping that records which thread reads it."""

        def __init__(self, data: dict[str, str]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> str:
            return self._data[key]

        def __len__(self) -> int:
            return len(self._data)

        def __iter__(self) -> Iterator[str]:
            read_idents.append(threading.get_ident())
            return iter(self._data)

    class probe(sqlalchemy_async[str, str]):
        def _populate_system_columns(self, rows: list[dict[str, Any]]) -> None:
            populate_idents.append(threading.get_ident())
            held_while_populating.append(engine_probe.held)

    async def main() -> None:
        storage = probe(
            url=f"sqlite+aiosqlite:///{tmp_path}/cases.db",
            value_column="text",
        )
        engine_probe.attach(storage.sa_engine)
        await storage.put_index(loop_affine({"a": "alpha"}))
        await storage.upsert_index(loop_affine({"b": "beta"}))
        # Replaces the row matching the filter ("b"), leaving "a" untouched.
        await storage.replace_where(Eq("text", "beta"), loop_affine({"c": "gamma"}))
        await storage.patch_index(upsert=loop_affine({"d": "delta"}), delete=["a"])
        assert await storage.get_index() == {"c": "gamma", "d": "delta"}
        await storage.close()

    asyncio.run(main())

    loop_thread = threading.get_ident()
    assert len(populate_idents) == 4
    assert all(ident != loop_thread for ident in populate_idents)
    assert read_idents
    assert all(ident == loop_thread for ident in read_idents)
    assert held_while_populating == [1, 0, 0, 0]


def test_sqlite_vec_host_table_reindex(tmp_path: Path) -> None:
    """Shadows are cbrkit-owned even when the data table is the host's.

    ``manage_schema=False`` governs only the main table, so ``vec0`` / FTS5
    must still be created, and ``reindex`` must backfill them from rows written
    outside cbrkit.  The host table deliberately carries a column named like the
    default ``vector_column``: the shadow's columns are a separate namespace, so
    the two must coexist untouched.  Embedding stays outside the write
    transaction here too, since SQLite allows only a single writer.
    """
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("sqlite_vec")
    pytest.importorskip("aiosqlite")

    import asyncio

    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import sqlite_vec_async

    held_while_embedding: list[int] = []
    probe = _engine_probe()

    def embed(texts: Sequence[str]) -> Sequence[NumpyArray]:
        held_while_embedding.append(probe.held)
        return _toy_embed(texts)

    metadata = sa.MetaData()
    table = sa.Table(
        "cases",
        metadata,
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        # Named like the default vector_column, which names a column of the
        # vec0 shadow: an independent namespace that must not touch this one.
        sa.Column("embedding", sa.Text, nullable=True),
    )

    async def main() -> tuple[int, int, int]:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/host.db")
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert(),
                [
                    {"id": "a", "text": "red sedan car", "embedding": "host-owned"},
                    {"id": "b", "text": "blue sky", "embedding": "host-owned"},
                ],
            )

        probe.attach(engine)
        storage = sqlite_vec_async[str, Any](
            engine=engine,
            table=table,
            value_column="text",
            vector_dim=6,
            index_type="hybrid",
            conversion_func=embed,
        )
        indexed = await storage.reindex(batch_size=1)
        # Writes through cbrkit keep the shadows in sync from then on.
        await storage.upsert_index(
            {"c": {"text": "red apple fruit", "embedding": "host-owned"}}
        )
        # The shadow vector never leaks into the host's own columns.
        assert [v for _, v in (await storage.get_index()).items()] == [
            {"text": "red sedan car", "embedding": "host-owned"},
            {"text": "blue sky", "embedding": "host-owned"},
            {"text": "red apple fruit", "embedding": "host-owned"},
        ]

        async with engine.connect() as conn:
            vecs = await conn.scalar(sa.text('SELECT count(*) FROM "cases_vec"')) or 0
            fts = await conn.scalar(sa.text('SELECT count(*) FROM "cases_fts"')) or 0

        await engine.dispose()

        return indexed, vecs, fts

    indexed, vecs, fts = asyncio.run(main())

    assert indexed == 2
    assert (vecs, fts) == (3, 3)
    # Two reindex pages plus the upsert, none of them holding a transaction.
    assert held_while_embedding == [0, 0, 0]


def test_pgvector_reembed_all_pages_outside_transactions(tmp_path: Path) -> None:
    """Re-embedding walks every page without ever holding a transaction open.

    ``reembed_all``'s body is dialect-agnostic, so a host-supplied table (no
    DDL, plain JSON column standing in for the vector) exercises it on SQLite.
    Regression: the whole rebuild ran inside one transaction, so a connection
    stayed checked out for every embedding batch of a potentially hours-long
    walk.
    """
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")
    pytest.importorskip("pgvector")

    import asyncio

    import numpy as np
    import sqlalchemy as sa
    from sqlalchemy.ext.asyncio import create_async_engine

    from cbrkit.indexable import pgvector_async

    held_while_embedding: list[int] = []
    batches: list[int] = []
    probe = _engine_probe()

    def embed(texts: Sequence[str]) -> Sequence[Any]:
        held_while_embedding.append(probe.held)
        batches.append(len(texts))
        return [np.asarray([float(len(t))]) for t in texts]

    class vec_json(sa.TypeDecorator[Any]):
        """JSON stand-in for ``VECTOR``, which SQLite does not have.

        Only the bind side needs a shim: the real pgvector types accept the
        ndarray :meth:`_populate_system_columns` produces, whereas ``JSON``
        cannot serialize one.
        """

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

    async def main() -> tuple[int, int, int]:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/cases.db")
        async with engine.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await conn.execute(
                table.insert(),
                [{"id": f"k{i}", "text": "x" * i, "vec": None} for i in range(5)],
            )

        probe.attach(engine)
        storage = pgvector_async[str, Any](
            engine=engine,
            table=table,
            value_column="text",
            pgvector_column="vec",
            index_type="dense",
            conversion_func=embed,
        )
        before = probe.commits
        walked = await storage.reembed_all(batch_size=2)
        paged_commits = probe.commits - before

        before = probe.commits
        await storage.reembed_all(batch_size=2, atomic=True)
        atomic_commits = probe.commits - before

        async with engine.connect() as conn:
            result = await conn.execute(sa.select(table.c.id, table.c.vec))
            stored = {key: vec for key, vec in result.all()}

        assert stored == {f"k{i}": [float(i)] for i in range(5)}
        await engine.dispose()

        return walked, paged_commits, atomic_commits

    walked, paged_commits, atomic_commits = asyncio.run(main())

    assert walked == 5
    # Every row reached, in pages of two, and nothing held while embedding.
    assert batches == [2, 2, 1] * 2
    assert held_while_embedding == [0] * 6
    # Paged commits once per page written; atomic buffers and commits once.
    assert atomic_commits == 1
    assert paged_commits > atomic_commits


def test_lancedb_patch_and_predicate_helpers(tmp_path: Path) -> None:
    """Exercise patch_index, native predicate helpers, and key escaping."""
    pytest.importorskip("lancedb")

    @dataclasses.dataclass
    class Doc:
        value: str
        source: str

    def _doc(key: str, value: str) -> Doc:
        return Doc(value=value, source=key.split("::", maxsplit=1)[0])

    storage = cbrkit.indexable.lancedb[str, Doc](
        uri=str(tmp_path),
        table_name="cases",
        index_type="sparse",
        model=Doc,
    )
    initial = {
        k: _doc(k, v)
        for k, v in {
            "doc-a::0": "alpha",
            "doc-a::1": "beta",
            "quote's::0": "gamma",
        }.items()
    }
    storage.put_index(initial)

    upsert = {
        k: _doc(k, v)
        for k, v in {
            "doc-a::0": "alpha updated",
            "doc-b::0": "delta",
        }.items()
    }
    storage.patch_index(upsert=upsert, delete=["quote's::0"])

    assert storage.index == {
        "doc-a::0": Doc("alpha updated", "doc-a"),
        "doc-a::1": Doc("beta", "doc-a"),
        "doc-b::0": Doc("delta", "doc-b"),
    }
    assert set(storage.keys_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert set(storage.delete_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert storage.index == {"doc-b::0": Doc("delta", "doc-b")}


def test_zvec_dense_fts_hybrid(tmp_path: Path) -> None:
    """End-to-end dense / FTS-sparse / hybrid retrieval over a real zvec collection."""
    pytest.importorskip("zvec")

    import gc

    cases = {"a": "red sedan car", "b": "blue sky", "c": "red apple fruit"}
    path = str(tmp_path / "cases")

    def _build() -> "cbrkit.indexable.zvec[str, str]":
        return cbrkit.indexable.zvec[str, str](
            path=path,
            collection_name="cases",
            index_type="hybrid",
            conversion_func=_toy_embed,
        )

    storage = _build()
    storage.put_index(cases)
    assert storage.has_index()
    assert dict(storage.index) == cases
    assert storage.search_limit() == 3

    # dense: "red car" is closest to "a" (red + car)
    dense = cbrkit.retrieval.indexable.zvec(storage, search_type="dense", limit=2)
    cb, sm = dense([({}, "red car")])[0]
    assert next(iter(sm)) == "a"
    assert cb["a"] == "red sedan car"

    # sparse: native FTS keyword "red" hits "a" and "c", not "b"
    sparse = cbrkit.retrieval.indexable.zvec(storage, search_type="sparse")
    _, sm = sparse([({}, "red")])[0]
    assert set(sm) == {"a", "c"}

    # hybrid: dense + FTS fused via RRF still surfaces the lexical "red" hits
    hybrid = cbrkit.retrieval.indexable.zvec(storage, search_type="hybrid", limit=5)
    _, sm = hybrid([({}, "red car")])[0]
    assert {"a", "c"} <= set(sm)

    # deletes propagate to the FTS index
    storage.delete_index(["a"])
    _, sm = sparse([({}, "red")])[0]
    assert set(sm) == {"c"}

    # reopening the persisted collection restores keys and serves FTS queries;
    # drop the handle first to release the read-write lock on the path
    storage._collection = None
    gc.collect()

    reopened = _build()
    assert reopened.has_index()
    assert reopened.search_limit() == 2
    refts = cbrkit.retrieval.indexable.zvec(reopened, search_type="sparse")
    _, sm = refts([({}, "red")])[0]
    assert set(sm) == {"c"}
