"""Retriever wrappers around storage backends in :mod:`cbrkit.indexable`."""

from ...helpers import optional_dependencies
from ._common import (
    _brute_force_dense_search,
    _normalize_results,
    _StorageIndexMixin,
    resolve_casebases,
)
from .embed import embed

with optional_dependencies():
    from .bm25 import bm25

with optional_dependencies():
    from .lancedb import lancedb

with optional_dependencies():
    from .chromadb import chromadb

with optional_dependencies():
    from .zvec import zvec

with optional_dependencies():
    from .postgresql import postgresql, postgresql_async

__all__ = [
    "resolve_casebases",
    "_normalize_results",
    "_brute_force_dense_search",
    "_StorageIndexMixin",
    "embed",
    "bm25",
    "lancedb",
    "chromadb",
    "zvec",
    "postgresql",
    "postgresql_async",
]
