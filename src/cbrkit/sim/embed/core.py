"""Core embedding combinators: build, cache, concat."""

import asyncio
import json
import sqlite3
from collections.abc import (
    Generator,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Literal,
    Protocol,
    override,
    runtime_checkable,
)

import numpy as np

from ...helpers import batchify, wait
from ...schema import HostFunction, HostRef, serializable
from ...typing import (
    AnySimFunc,
    BatchSimFunc,
    EmbeddingFunc,
    NumpyArray,
    SimSeq,
)
from .metrics import VECTOR_METRICS


@runtime_checkable
class _HasQueries[V, E](Protocol):
    """An embedding function with another function for queries, such as an `EmbeddingCache`."""

    @property
    def queries(self) -> EmbeddingFunc[V, E]: ...


def for_queries[V, E](embedder: EmbeddingFunc[V, E]) -> EmbeddingFunc[V, E]:
    """The function that embeds queries, which keeps them out of the store if `embedder` is an `EmbeddingCache`.

    Args:
        embedder: Embeds the queries, possibly an `EmbeddingCache` or an `Embedder` bound to one.
    """
    return embedder.queries if isinstance(embedder, _HasQueries) else embedder


async def embed_pairs[V, E](
    embedder: EmbeddingFunc[V, E],
    query_embedder: EmbeddingFunc[V, E] | None,
    case_texts: Sequence[V],
    query_texts: Sequence[V],
) -> tuple[Sequence[E], Sequence[E]]:
    """Embed case and query texts, returning their vector sequences.

    The queries are embedded with `for_queries` of `query_embedder`, or of `embedder` if it is None.
    If that is `embedder` itself, one batched call embeds both,
    otherwise the queries are embedded concurrently.
    Every distinct text is embedded only once per function.
    """
    queries = for_queries(query_embedder or embedder)

    if queries is embedder:
        vectors = await _embed_unique(embedder, [*case_texts, *query_texts])

        return vectors[: len(case_texts)], vectors[len(case_texts) :]

    return await asyncio.gather(
        _embed_unique(embedder, case_texts), _embed_unique(queries, query_texts)
    )


async def _embed_unique[V, E](
    embedder: EmbeddingFunc[V, E], values: Sequence[V]
) -> Sequence[E]:
    """Embeds every distinct value once and maps the vectors back to all values."""
    unique = list(dict.fromkeys(values))

    if len(unique) == len(values):
        return await embedder(values)

    vectors = dict(zip(unique, await embedder(unique), strict=True))

    return [vectors[value] for value in values]


@dataclass(slots=True, frozen=True)
class Embedder[E = Any](HostFunction[EmbeddingFunc[str, E]], EmbeddingFunc[str, E]):
    """Embedding function of the host, serialized by its name only, see `cbrkit.schema.HostFunction`.

    The `embedders` of the providers section of `cbrkit.spec` name the embedding functions.
    It is generic over the embeddings `E` of the bound function, such as `NumpyArray` or `SparseVector`.
    """

    @property
    def queries(self) -> EmbeddingFunc[str, E]:
        """The `for_queries` function of the bound function, itself if that has none."""
        func = self.bound()

        return func.queries if isinstance(func, _HasQueries) else self

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[E]:
        return await self.bound()(values)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Embedding[E](BatchSimFunc[str, float]):
    """Semantic similarity of the embeddings of two texts.

    Generic over the embedding type `E` of the metric,
    dense (`NumpyArray`) or sparse (`SparseVector`) embeddings.
    The measure is synchronous like every measure and waits for the asynchronous embedder
    with `cbrkit.helpers.wait`, on the event loop of `cbrkit.cycle.run_async` inside a run.

    Args:
        embedder: Embeds the texts, and the queries if `query_embedder` is None.
        metric: Similarity of two embeddings, such as `Cosine` for dense and `SparseCosine` for sparse ones.
        query_embedder: Embeds the queries, if they need another model or input type.
    """

    embedder: HostRef[Embedder[E]]
    metric: Annotated[AnySimFunc[E, float], VECTOR_METRICS]
    query_embedder: HostRef[Embedder[E]] | None = None
    kind: Literal["embedding"] = field(default="embedding", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq[float]:
        if not batches:
            return []

        case_texts, query_texts = zip(*batches, strict=True)
        case_vecs, query_vecs = wait(
            embed_pairs(self.embedder, self.query_embedder, case_texts, query_texts)
        )

        return batchify(self.metric)(list(zip(case_vecs, query_vecs, strict=True)))


class _Keyed(Protocol):
    """Embeddings by text that `dict` accepts, such as a mapping."""

    def keys(self) -> Iterable[str]: ...

    def __getitem__(self, text: str, /) -> NumpyArray: ...


def _quote(identifier: str) -> str:
    """Quotes an SQL identifier, so any table name is safe in a statement."""
    return '"' + identifier.replace('"', '""') + '"'


def _create_table(con: sqlite3.Connection, table: str) -> None:
    """Creates the table unless it exists, so a database that was deleted or is new works, and reads issue no DDL."""
    if con.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone():
        return

    # the mode persists, so losing the race to switch a new database to it is harmless
    with suppress(sqlite3.OperationalError):
        con.execute("PRAGMA journal_mode=WAL")

    with con:
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {_quote(table)}"
            " (model TEXT NOT NULL, text TEXT NOT NULL, vector BLOB NOT NULL,"
            " PRIMARY KEY (model, text)) STRICT"
        )


@dataclass(slots=True, frozen=True)
class SQLiteStore(MutableMapping[str, NumpyArray]):
    """Embeddings of one model by text in a table of a SQLite database, a persistent store of an `EmbeddingCache`.

    Every access opens the database, which creates the file and the table when they are missing,
    so creating the store does no I/O, and `update` writes many embeddings in one transaction.
    The database is in WAL mode and waits for the locks of other processes, so they can share it.
    Several models can share one table without mixing their vectors,
    but the vectors are keyed by the model identifier only,
    so another model served under the same identifier needs another store or table.

    Args:
        path: Path of the SQLite database.
        model: Identifier of the embedding model.
        table: Table of the database.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> path = Path(tempfile.mkdtemp()) / "cache.sqlite"
        >>> SQLiteStore(path, "model").update({"text": np.ones(2)})
        >>> SQLiteStore(path, "model")["text"].tolist(), len(SQLiteStore(path, "other"))
        ([1.0, 1.0], 0)
    """

    path: Path
    model: str
    table: str = "embeddings"

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, timeout=30)

        try:
            _create_table(con, self.table)

            with con:
                yield con
        finally:
            con.close()

    def __getitem__(self, text: str) -> NumpyArray:
        with self._connect() as con:
            row = con.execute(
                f"SELECT vector FROM {_quote(self.table)} WHERE model = ? AND text = ?",
                (self.model, text),
            ).fetchone()

        if row is None:
            raise KeyError(text)

        return np.frombuffer(row[0], dtype=np.float64)

    def get_many(self, texts: Iterable[str], /) -> dict[str, NumpyArray]:
        """Reads the stored embeddings of the given texts in one query, skipping missing ones."""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT text, vector FROM {_quote(self.table)}"
                " WHERE model = ? AND text IN (SELECT value FROM json_each(?))",
                (self.model, json.dumps(list(texts))),
            ).fetchall()

        return {text: np.frombuffer(vector, dtype=np.float64) for text, vector in rows}

    def __setitem__(self, text: str, vector: NumpyArray) -> None:
        self.update({text: vector})

    @override
    def update(
        self,
        other: _Keyed | Iterable[tuple[str, NumpyArray]] = (),
        /,
        **kwargs: NumpyArray,
    ) -> None:
        """Writes embeddings in one transaction."""
        vectors = dict(other, **kwargs)

        with self._connect() as con:
            con.executemany(
                f"INSERT OR REPLACE INTO {_quote(self.table)} (model, text, vector) VALUES (?, ?, ?)",
                [
                    (self.model, text, np.asarray(vector, dtype=np.float64).tobytes())
                    for text, vector in vectors.items()
                ],
            )

    def __delitem__(self, text: str) -> None:
        with self._connect() as con:
            deleted = con.execute(
                f"DELETE FROM {_quote(self.table)} WHERE model = ? AND text = ?",
                (self.model, text),
            ).rowcount

        if not deleted:
            raise KeyError(text)

    def __iter__(self) -> Iterator[str]:
        with self._connect() as con:
            texts = [
                row[0]
                for row in con.execute(
                    f"SELECT text FROM {_quote(self.table)} WHERE model = ?",
                    (self.model,),
                )
            ]

        return iter(texts)

    def __len__(self) -> int:
        with self._connect() as con:
            return con.execute(
                f"SELECT COUNT(*) FROM {_quote(self.table)} WHERE model = ?",
                (self.model,),
            ).fetchone()[0]


@dataclass(slots=True, frozen=True)
class Scratch[E](MutableMapping[str, E]):
    """Embeddings that read through to a store but write only to a bounded memory, least recently used first out.

    As the store of an `EmbeddingCache`, it keeps new texts, such as queries, out of a persistent store.

    Args:
        base: Embeddings by text that are only read, such as a `SQLiteStore`.
        max_size: Most embeddings kept in memory.
    """

    base: Mapping[str, E] = field(repr=False)
    max_size: int = 1024
    _memory: dict[str, E] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @override
    def __getitem__(self, text: str, /) -> E:
        if text in self._memory:
            self._memory[text] = self._memory.pop(text)

            return self._memory[text]

        return self.base[text]

    def get_many(self, texts: Sequence[str], /) -> dict[str, E]:
        """Reads the known embeddings of the given texts, the missing ones from the base in one go."""
        known = {text: self[text] for text in texts if text in self._memory}
        known |= _get_many(self.base, [text for text in texts if text not in known])

        return known

    @override
    def __setitem__(self, text: str, vector: E, /) -> None:
        self._memory.pop(text, None)
        self._memory[text] = vector

        if len(self._memory) > self.max_size:
            del self._memory[next(iter(self._memory))]

    @override
    def __delitem__(self, text: str, /) -> None:
        del self._memory[text]

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._memory)

    @override
    def __len__(self) -> int:
        return len(self._memory)


@runtime_checkable
class _ManyGetter[E](Protocol):
    """A store that reads the embeddings of many texts at once, such as a `SQLiteStore`."""

    def get_many(self, texts: Sequence[str], /) -> dict[str, E]: ...


def _get_many[E](store: Mapping[str, E], texts: Sequence[str]) -> dict[str, E]:
    """Reads the stored embeddings of the given texts, skipping missing ones, in one query where the store supports it."""
    if isinstance(store, _ManyGetter):
        return store.get_many(texts)

    return {text: vector for text in texts if (vector := store.get(text)) is not None}


@dataclass(slots=True, frozen=True, kw_only=True)
class EmbeddingCache[E](EmbeddingFunc[str, E]):
    """Keeps the embeddings of texts in an explicit store to avoid computing them again.

    Every new text is embedded once and written to the store, so calling the cache prewarms it.
    The `queries` of the cache read the store but keep new texts only in a bounded memory,
    so `for_queries` neither grows a persistent store with queries nor discloses them.
    The store is read and written in a worker thread, since it may be a file such as a `SQLiteStore`.
    It is generic over the embeddings `E` of the store, such as `NumpyArray` or `SparseVector` in a dict.

    Args:
        embedder: Embeds the texts that are not in the store yet.
        store: Embeddings by text, which the caller owns and may share or persist, such as a `SQLiteStore`.

    Examples:
        >>> import asyncio
        >>> calls = []
        >>> async def embed(texts):
        ...     calls.append(list(texts))
        ...     return [np.full(2, len(text), dtype=np.float64) for text in texts]
        >>> store = {}
        >>> cache = EmbeddingCache(embedder=embed, store=store)
        >>> [vector.tolist() for vector in asyncio.run(cache(["a", "bb", "a"]))]
        [[1.0, 1.0], [2.0, 2.0], [1.0, 1.0]]
        >>> _ = asyncio.run(cache(["bb"]))
        >>> _ = asyncio.run(for_queries(cache)(["bb", "q", "q"]))
        >>> _ = asyncio.run(for_queries(cache)(["q"]))
        >>> calls, list(store)
        ([['a', 'bb'], ['q']], ['a', 'bb'])
    """

    embedder: EmbeddingFunc[str, E]
    store: MutableMapping[str, E] = field(repr=False)
    queries: EmbeddingFunc[str, E] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "queries",
            self
            if isinstance(self.store, Scratch)
            else EmbeddingCache(embedder=self.embedder, store=Scratch(self.store)),
        )

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[E]:
        unique = list(dict.fromkeys(values))
        known = await asyncio.to_thread(self._lookup, unique)
        missing = [text for text in unique if text not in known]

        if missing:
            vectors = dict(zip(missing, await self.embedder(missing), strict=True))
            known |= vectors
            await asyncio.to_thread(self.store.update, vectors)

        return [known[text] for text in values]

    def _lookup(self, texts: Sequence[str]) -> dict[str, E]:
        if not texts:
            return {}

        return _get_many(self.store, texts)


@dataclass(slots=True, frozen=True, kw_only=True)
class Concat[V](EmbeddingFunc[V, NumpyArray]):
    """Concatenated embeddings of multiple models, which embed concurrently.

    Args:
        embedders: Embedding functions whose vectors are concatenated in order.
    """

    embedders: Sequence[EmbeddingFunc[V, NumpyArray]]

    @override
    async def __call__(self, values: Sequence[V], /) -> Sequence[NumpyArray]:
        nested_vecs = await asyncio.gather(*(func(values) for func in self.embedders))

        return [np.concatenate(vecs, axis=0) for vecs in zip(*nested_vecs, strict=True)]


__all__ = [
    "Concat",
    "Embedder",
    "Embedding",
    "EmbeddingCache",
    "SQLiteStore",
    "Scratch",
    "embed_pairs",
    "for_queries",
]
