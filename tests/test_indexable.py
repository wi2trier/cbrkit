"""Tests for indexable storage backends."""

from pathlib import Path

import pytest

import cbrkit


def test_lancedb_patch_and_predicate_helpers(tmp_path: Path) -> None:
    """Exercise patch_index, native predicate helpers, and key escaping."""
    pytest.importorskip("lancedb")

    def metadata(key: str, _value: str) -> dict[str, str]:
        return {"source": key.split("::", maxsplit=1)[0]}

    storage = cbrkit.indexable.lancedb[str](
        uri=str(tmp_path),
        table_name="cases",
        index_type="sparse",
        metadata_func=metadata,
    )
    storage.put_index(
        {
            "doc-a::0": "alpha",
            "doc-a::1": "beta",
            "quote's::0": "gamma",
        }
    )

    retriever = cbrkit.retrieval.lancedb(storage=storage, search_type="sparse")
    retriever.patch_index(
        upsert={"doc-a::0": "alpha updated", "doc-b::0": "delta"},
        delete=["quote's::0"],
    )

    assert retriever.index == {
        "doc-a::0": "alpha updated",
        "doc-a::1": "beta",
        "doc-b::0": "delta",
    }
    assert set(storage.keys_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert set(storage.delete_where("source = 'doc-a'")) == {"doc-a::0", "doc-a::1"}
    assert storage.index == {"doc-b::0": "delta"}
