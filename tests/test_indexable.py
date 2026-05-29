"""Tests for indexable storage backends."""

import dataclasses
from pathlib import Path

import pytest

import cbrkit


def _field_set(cls: type) -> set[str]:
    """Public dataclass field names (excludes private ``_``-prefixed slots)."""
    return {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}


def test_sqlalchemy_storage_sync_async_field_parity() -> None:
    """Sync `sqlalchemy` storage must expose every config the async one does.

    The sync facade restricts itself to a URL-only construction shape, so
    `url` / `engine` / `table` / `metadata` / `manage_schema` are
    legitimately async-only.  Any other field on the async class must be
    mirrored on the sync facade or we ship divergent defaults.
    """
    pytest.importorskip("sqlalchemy")

    from cbrkit.indexable.sqlalchemy import sqlalchemy, sqlalchemy_async

    async_only = {"url", "engine", "table", "metadata"}
    missing = _field_set(sqlalchemy_async) - _field_set(sqlalchemy) - async_only
    assert not missing, (
        f"sqlalchemy sync facade is missing fields present on sqlalchemy_async: "
        f"{missing}"
    )


def test_pgvector_storage_sync_async_field_parity() -> None:
    """Sync `pgvector` storage must expose every config the async one does."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pgvector")

    from cbrkit.indexable.pgvector import pgvector, pgvector_async

    async_only = {"url", "engine", "table", "metadata"}
    missing = _field_set(pgvector_async) - _field_set(pgvector) - async_only
    assert not missing, (
        f"pgvector sync facade is missing fields present on pgvector_async: {missing}"
    )


def test_pgvector_retriever_sync_async_field_parity() -> None:
    """Sync `pgvector` retriever must mirror the async retriever's fields."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pgvector")

    from cbrkit.retrieval.indexable.pgvector import pgvector, pgvector_async

    missing = _field_set(pgvector_async) - _field_set(pgvector)
    assert not missing, (
        f"pgvector sync retriever is missing fields present on pgvector_async: "
        f"{missing}"
    )


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
    initial = {k: _doc(k, v) for k, v in {
        "doc-a::0": "alpha",
        "doc-a::1": "beta",
        "quote's::0": "gamma",
    }.items()}
    storage.put_index(initial)

    upsert = {k: _doc(k, v) for k, v in {
        "doc-a::0": "alpha updated",
        "doc-b::0": "delta",
    }.items()}
    storage.patch_index(upsert=upsert, delete=["quote's::0"])

    assert storage.index == {
        "doc-a::0": Doc("alpha updated", "doc-a"),
        "doc-a::1": Doc("beta", "doc-a"),
        "doc-b::0": Doc("delta", "doc-b"),
    }
    assert set(storage.keys_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert set(storage.delete_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert storage.index == {"doc-b::0": Doc("delta", "doc-b")}
