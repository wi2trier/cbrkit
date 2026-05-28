"""Retriever wrappers around storage backends in :mod:`cbrkit.indexable`."""

from ...helpers import optional_dependencies
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
    from .pgvector import pgvector, pgvector_async

__all__ = [
    "embed",
    "bm25",
    "lancedb",
    "chromadb",
    "zvec",
    "pgvector",
    "pgvector_async",
]
