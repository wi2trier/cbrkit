"""Storage backends for indexable retrieval.

Each storage implements `cbrkit.typing.Index` asynchronously, since it does I/O.
A storage is configuration, it does no I/O until it is used,
and the host opens and closes the resources it takes:
the SQLAlchemy storages take an `AsyncEngine`, `LanceDB` takes a `lancedb.AsyncConnection`,
and `Zvec` opens the collection at its path per operation.
That resource is the only positional argument, the others are keywords,
and the storages share their defaults, such as the table `"cases"` with the columns `"key"` and `"value"`.
The SQLAlchemy storages create the schema objects that cbrkit owns with `create`,
`reflect` reads an existing table for them,
and `reindex` rebuilds their search index from the stored rows, such as after the embedder changed.
Their required `scope` restricts them to the rows matching an expression, so one table backs many storages, such as one per user,
`narrow` restricts one further, and only an indexer that writes every scope's rows holds one of `Unscoped.TABLE`.
Storages with a search index are searched by their retrievers, such as `PgvectorRetriever(storage=...)`,
so several retrievers with different settings can share one storage,
and such a retriever comes first in a run without a source.
A retriever runs a `Dense`, `Sparse` or `Hybrid` search, by default the `default_search` of the `IndexType` of its storage.
Any storage is also the source of a run, `cbrkit.cycle.run_query(storage, query, ...)` reads its current cases once,
for instance those that `InMemory` keeps in memory.
Any of them except `LanceDB`, which implements no `cbrkit.typing.Storage`,
stores the changes of `cbrkit.retain.Retainer` and `cbrkit.maintain.Maintainer` with `cbrkit.model.replan`.

The value type `V` follows the schema source,
and the embeddable text is named by `value_column`:
- Plain text (`V = str`, the default without a `model`) is stored under the text column and read back as a string.
  The SQLAlchemy storages keep it in a table they build or in a `table` of the host,
  whose other columns keep their defaults on writes and may be tested by the `scope`.
- A typed model is given as `model`, a dataclass or Pydantic model for `LanceDB` and `Zvec`,
  or a SQLAlchemy mapped class for `SQLAlchemy`, `Pgvector` and `SQLiteVec` (declare its vector column
  with `PGVECTOR` or `HALFVEC`), and reads reconstruct model instances.
- A mapping (`V = Mapping[str, Any]`) needs a `table` of the host and no `value_column`,
  so only `SQLAlchemy` stores it, for instance in an existing table read with `reflect`.

The retrievers of the storages search by the text column and return `Casebase[K, str]`,
so a retrieve and retain round trip lines up only for plain text.
For typed values, retrieve from `get_index()` with a value-based retriever such as `cbrkit.retrieve.Retriever`,
or look up the full rows by key after text retrieval.

Example:
    >>> import asyncio
    >>> import cbrkit
    >>> from sqlalchemy.ext.asyncio import create_async_engine
    >>> async def main() -> tuple[str, ...]:
    ...     engine = create_async_engine("sqlite+aiosqlite://")
    ...     storage = cbrkit.indexable.SQLiteVec[str, str](
    ...         engine,
    ...         value_column="text",
    ...         index_type="sparse",
    ...         scope=cbrkit.indexable.Unscoped.TABLE,
    ...     )
    ...     await storage.create()
    ...     await storage.put_index({"a": "hello world", "b": "foo bar"})
    ...     retriever = cbrkit.indexable.SQLiteVecRetriever(storage=storage)
    ...     result = await cbrkit.cycle.run_query_async(None, "hello", retriever)
    ...     await engine.dispose()
    ...     return result.ranking
    >>> asyncio.run(main())
    ('a',)
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports

if TYPE_CHECKING:
    from ._lancedb import LanceDB, LanceDBRetriever
    from ._memory import InMemory
    from ._pgvector import (
        HALFVEC,
        PGVECTOR,
        TSVECTOR,
        Pgvector,
        PgvectorRetriever,
        tsvector_computed,
    )
    from ._retriever import (
        Dense,
        Hybrid,
        IndexType,
        Search,
        Sparse,
        default_search,
    )
    from ._sqlalchemy import SQLAlchemy, StorageScope, Unscoped, reflect
    from ._sqlite_vec import SQLiteVec, SQLiteVecRetriever
    from ._zvec import Zvec, ZvecRetriever

__all__ = [
    "HALFVEC",
    "PGVECTOR",
    "TSVECTOR",
    "Dense",
    "Hybrid",
    "InMemory",
    "IndexType",
    "LanceDB",
    "LanceDBRetriever",
    "Pgvector",
    "PgvectorRetriever",
    "SQLAlchemy",
    "SQLiteVec",
    "SQLiteVecRetriever",
    "Search",
    "Sparse",
    "StorageScope",
    "Unscoped",
    "Zvec",
    "ZvecRetriever",
    "default_search",
    "reflect",
    "tsvector_computed",
]

lazy_exports(__name__)
