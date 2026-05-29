"""Tests for indexable storage backends."""

import dataclasses
from pathlib import Path

import pytest

import cbrkit


def _toy_embed(texts: list[str]):
    """Deterministic bag-of-keywords embedder for sqlite_vec tests."""
    import numpy as np

    vocab = ("red", "blue", "green", "car", "sky", "fruit")
    rows = []
    for text in texts:
        lowered = text.lower()
        vec = np.array([float(word in lowered) for word in vocab], dtype=np.float32)
        if not vec.any():
            vec[:] = 1.0
        rows.append(vec)
    return np.asarray(rows)


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
