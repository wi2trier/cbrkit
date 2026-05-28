"""Tests for indexable storage backends."""

import dataclasses
from pathlib import Path

import pytest

import cbrkit


def _field_set(cls: type) -> set[str]:
    """Public dataclass field names (excludes private ``_``-prefixed slots)."""
    return {f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")}


def test_postgresql_storage_sync_async_field_parity() -> None:
    """Sync `postgresql` storage must expose every config the async one does.

    The sync facade restricts itself to a URL-only construction shape, so
    `url` / `engine` / `table` / `metadata` / `manage_schema` are
    legitimately async-only.  Any other field on the async class must be
    mirrored on the sync facade or we ship divergent defaults.
    """
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pgvector")

    from cbrkit.indexable.postgresql import postgresql, postgresql_async

    async_only = {"url", "engine", "table", "metadata", "manage_schema"}
    missing = _field_set(postgresql_async) - _field_set(postgresql) - async_only
    assert not missing, (
        f"postgresql sync facade is missing fields present on postgresql_async: {missing}"
    )


def test_postgresql_retriever_sync_async_field_parity() -> None:
    """Sync `postgresql` retriever must mirror the async retriever's fields."""
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("pgvector")

    from cbrkit.retrieval.indexable.postgresql import postgresql, postgresql_async

    missing = _field_set(postgresql_async) - _field_set(postgresql)
    assert not missing, (
        f"postgresql sync retriever is missing fields present on postgresql_async: {missing}"
    )


def test_lancedb_patch_and_predicate_helpers(tmp_path: Path) -> None:
    """Exercise patch_index, native predicate helpers, and key escaping."""
    pytest.importorskip("lancedb")

    def _source(keys: list[str]) -> dict[str, dict[str, str]]:
        return {k: {"source": k.split("::", maxsplit=1)[0]} for k in keys}

    storage = cbrkit.indexable.lancedb[str](
        uri=str(tmp_path),
        table_name="cases",
        index_type="sparse",
    )
    initial = {
        "doc-a::0": "alpha",
        "doc-a::1": "beta",
        "quote's::0": "gamma",
    }
    storage.put_index(initial, metadata=_source(list(initial)))

    retriever = cbrkit.retrieval.lancedb(storage=storage, search_type="sparse")
    upsert = {"doc-a::0": "alpha updated", "doc-b::0": "delta"}
    storage.patch_index(
        upsert=upsert,
        delete=["quote's::0"],
        metadata=_source(list(upsert)),
    )

    assert retriever.index == {
        "doc-a::0": "alpha updated",
        "doc-a::1": "beta",
        "doc-b::0": "delta",
    }
    assert set(storage.keys_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert set(storage.delete_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert storage.index == {"doc-b::0": "delta"}
