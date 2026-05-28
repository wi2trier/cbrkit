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
from ._common import (
    PG_METRICS,
    _PgMetric,
    _compute_index_diff,
    _normalize_patch_keys,
    _sql_in_clause,
    _sql_literal,
)

with optional_dependencies():
    from .lancedb import lancedb

with optional_dependencies():
    from .chromadb import chromadb

with optional_dependencies():
    from .zvec import zvec

with optional_dependencies():
    from .postgresql import postgresql, postgresql_async, postgresql_table

__all__ = [
    "chromadb",
    "lancedb",
    "postgresql",
    "postgresql_async",
    "postgresql_table",
    "zvec",
    "_compute_index_diff",
    "_normalize_patch_keys",
    "_sql_literal",
    "_sql_in_clause",
    "_PgMetric",
    "PG_METRICS",
]
