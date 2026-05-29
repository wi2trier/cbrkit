"""Pure storage backends for indexable retrieval.

This module provides storage classes that implement
:class:`~cbrkit.typing.IndexableFunc` without any retrieval logic.
Each backend manages database connections, data ingestion, and index
maintenance.  Retriever wrappers in :mod:`cbrkit.retrieval` consume
these storage instances to perform search queries.

Example:
    >>> import tempfile  # doctest: +SKIP
    >>> storage = lancedb(  # doctest: +SKIP
    ...     uri=tempfile.mkdtemp(),
    ...     table_name="cases",
    ...     index_type="sparse",
    ... )
    >>> storage.put_index({0: "hello world", 1: "foo bar"})  # doctest: +SKIP
    >>> storage.has_index()  # doctest: +SKIP
    True
"""

from ..helpers import optional_dependencies

with optional_dependencies():
    from .lancedb import lancedb

with optional_dependencies():
    from .chromadb import chromadb

with optional_dependencies():
    from .zvec import zvec

with optional_dependencies():
    from .sqlalchemy import sqlalchemy, sqlalchemy_async

with optional_dependencies():
    from .pgvector import HALFVEC, PGVECTOR, TSVECTOR, pgvector, pgvector_async

with optional_dependencies():
    from .sqlite_vec import sqlite_vec, sqlite_vec_async

__all__ = [
    "HALFVEC",
    "PGVECTOR",
    "TSVECTOR",
    "chromadb",
    "lancedb",
    "pgvector",
    "pgvector_async",
    "sqlalchemy",
    "sqlalchemy_async",
    "sqlite_vec",
    "sqlite_vec_async",
    "zvec",
]
