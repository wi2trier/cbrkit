"""Tests for indexable storage backends."""

import dataclasses
from collections.abc import Sequence
from pathlib import Path

import pytest

import cbrkit
from cbrkit.typing import NumpyArray


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
    """System-column population must run in a worker thread, not on the loop.

    Regression: pgvector's ``conversion_func`` ran synchronously inside
    ``replace_where`` / ``put_index``, freezing the host application's
    event loop for the duration of the embedding batch.  Caller-owned
    *data* must still be read on the loop thread: mappings and ORM/model
    values may be thread-affine, so only the population of the encoded
    plain dicts may be offloaded.
    """
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("aiosqlite")

    import asyncio
    import threading
    from collections.abc import Iterator, Mapping
    from typing import Any

    from cbrkit.filter import Eq
    from cbrkit.indexable import sqlalchemy_async

    populate_idents: list[int] = []
    read_idents: list[int] = []

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

    async def main() -> None:
        storage = probe(
            url=f"sqlite+aiosqlite:///{tmp_path}/cases.db",
            value_column="text",
        )
        await storage.put_index(loop_affine({"a": "alpha"}))
        await storage.upsert_index(loop_affine({"b": "beta"}))
        await storage.replace_where(Eq("text", "beta"), loop_affine({"c": "gamma"}))
        await storage.close()

    asyncio.run(main())

    loop_thread = threading.get_ident()
    assert len(populate_idents) == 3
    assert all(ident != loop_thread for ident in populate_idents)
    assert read_idents
    assert all(ident == loop_thread for ident in read_idents)


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
