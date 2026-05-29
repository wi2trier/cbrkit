"""Core embedding combinators: build, cache, concat."""

import sqlite3
from collections import ChainMap
from collections.abc import Collection, Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import numpy as np

from ...helpers import batchify_conversion, batchify_sim
from ...indexable._common import _normalize_patch_keys
from ...typing import (
    AnyConversionFunc,
    AnySimFunc,
    BatchConversionFunc,
    BatchSimFunc,
    FilePath,
    Float,
    IndexableFunc,
    NumpyArray,
    SimSeq,
)
from .metrics import default_score_func


def embed_pairs[V, E](
    conversion_func: BatchConversionFunc[V, E],
    query_conversion_func: BatchConversionFunc[V, E] | None,
    case_texts: Sequence[V],
    query_texts: Sequence[V],
) -> tuple[Sequence[E], Sequence[E]]:
    """Embed case and query texts, returning their vector sequences.

    When *query_conversion_func* is ``None`` both sides share
    *conversion_func* and are embedded in a single batched call, then
    split apart — saving one round-trip over embedding them separately.
    """
    if query_conversion_func is not None:
        return conversion_func(case_texts), query_conversion_func(query_texts)

    all_vecs = conversion_func([*case_texts, *query_texts])
    split = len(case_texts)
    return all_vecs[:split], all_vecs[split:]


@dataclass(slots=True, init=False)
class build[V, E, S: Float](BatchSimFunc[V, S]):
    """Embedding-based semantic similarity.

    Generic over embedding type `E`, supporting both dense
    (`NumpyArray`) and sparse (`SparseVector`) embeddings.

    Args:
        conversion_func: Embedding function producing type `E`.
        sim_func: Similarity score function for type `E`.
        query_conversion_func: Optional query embedding function.
    """

    conversion_func: BatchConversionFunc[V, E]
    sim_func: BatchSimFunc[E, S]
    query_conversion_func: BatchConversionFunc[V, E] | None

    def __init__(
        self,
        conversion_func: AnyConversionFunc[V, E],
        sim_func: AnySimFunc[E, S] = default_score_func,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
        query_conversion_func: AnyConversionFunc[V, E] | None = None,
    ):
        self.conversion_func = batchify_conversion(conversion_func)
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = (
            batchify_conversion(query_conversion_func)
            if query_conversion_func
            else None
        )

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[S]:
        if not batches:
            return []

        case_texts, query_texts = zip(*batches, strict=True)
        case_vecs, query_vecs = embed_pairs(
            self.conversion_func, self.query_conversion_func, case_texts, query_texts
        )

        return self.sim_func(list(zip(case_vecs, query_vecs, strict=True)))


@dataclass(slots=True, init=False)
class cache(
    BatchConversionFunc[str, NumpyArray],
    IndexableFunc[Collection[str]],
):
    """Embedding cache with optional SQLite persistence.

    Cached vectors are keyed by text only — the embedding model that
    produced them is not recorded.  Reusing a `path`/`table` from a
    previous run while pointing :paramref:`func` at a different model
    silently returns the old model's vectors for previously-seen
    texts, mixing them with new-model vectors for new texts.  Delete
    the SQLite database (or use a fresh `table`) when changing models.
    """

    func: BatchConversionFunc[str, NumpyArray] | None
    path: Path | None
    table: str | None
    store: MutableMapping[str, NumpyArray] = field(repr=False)

    def __init__(
        self,
        func: AnyConversionFunc[str, NumpyArray] | None,
        path: FilePath | None = None,
        table: str | None = None,
    ):
        self.func = batchify_conversion(func) if func is not None else None
        self.path = Path(path) if isinstance(path, str) else path
        self.table = table
        self.store = {}

        if self.path is not None:
            if self.table is None:
                raise ValueError("Table name must be specified for disk cache")

            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.connect() as con:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.table}" (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL UNIQUE,
                        vector BLOB NOT NULL
                    ) STRICT
                """)
                con.commit()

                cur = con.execute(f'SELECT text, vector FROM "{self.table}"')

                for text, vector_blob in cur:
                    self.store[text] = np.frombuffer(vector_blob, dtype=np.float64)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Open a context-managed SQLite connection to the cache database."""
        if self.path is None:
            raise ValueError("Path must be set to use the cache")

        con = sqlite3.connect(self.path, autocommit=False)

        try:
            yield con
        finally:
            con.close()

    def _compute_vecs(self, texts: Sequence[str] | set[str]) -> dict[str, NumpyArray]:
        new_texts = [text for text in texts if text not in self.store]

        if not new_texts:
            return {}

        if self.func is None:
            raise ValueError("Conversion function is required for computing embeddings")

        new_vecs = self.func(new_texts)

        return dict(zip(new_texts, new_vecs, strict=True))

    @override
    def has_index(self) -> bool:
        """Return whether the cache contains any entries."""
        return bool(self.store)

    @property
    @override
    def index(self) -> Collection[str]:
        """Return the indexed texts."""
        return self.store.keys()

    @override
    def put_index(
        self,
        data: Collection[str],
    ) -> None:
        """Replace the embedding cache contents with *data*.

        On first call the cache is populated from scratch.  On
        subsequent calls only stale or new entries are removed/added,
        so unchanged texts skip re-embedding.
        """
        data_set = set(data)

        if not data_set:
            self.patch_index(delete=list(self.store.keys()))
            return

        stale_keys = set(self.store.keys()) - data_set
        new_keys = data_set - set(self.store.keys())

        if not stale_keys and not new_keys:
            return

        self.patch_index(
            upsert=new_keys or None,
            delete=stale_keys or None,
        )

    @override
    def upsert_index(
        self,
        data: Collection[str],
    ) -> None:
        """Add new embeddings to the cache.

        Texts already present in the cache are skipped.
        """
        new_vecs = self._compute_vecs(set(data))
        self.store.update(new_vecs)

        if self.path is not None and self.table is not None and new_vecs:
            with self.connect() as con:
                con.executemany(
                    f'INSERT INTO "{self.table}" (text, vector) VALUES(?, ?)',
                    [
                        (text, vector.astype(np.float64).tobytes())
                        for text, vector in new_vecs.items()
                    ],
                )
                con.commit()

    @override
    def delete_index(
        self,
        data: Collection[str],
    ) -> None:
        """Remove specific entries from the cache and SQLite."""
        if not data:
            return

        for text in data:
            self.store.pop(text, None)

        if self.path is not None and self.table is not None:
            with self.connect() as con:
                con.executemany(
                    f'DELETE FROM "{self.table}" WHERE text = ?',
                    [(text,) for text in data],
                )
                con.commit()

    @override
    def patch_index(
        self,
        upsert: Collection[str] | None = None,
        delete: Collection[str] | None = None,
    ) -> None:
        """Apply cache insertions and deletions."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        upsert_keys, delete_keys = normalized
        new_vecs = self._compute_vecs(upsert_keys)

        for text in delete_keys:
            self.store.pop(text, None)

        self.store.update(new_vecs)

        if self.path is not None and self.table is not None:
            with self.connect() as con:
                if delete_keys:
                    con.executemany(
                        f'DELETE FROM "{self.table}" WHERE text = ?',
                        [(text,) for text in delete_keys],
                    )

                if new_vecs:
                    con.executemany(
                        f'INSERT INTO "{self.table}" (text, vector) VALUES(?, ?)',
                        [
                            (text, vector.astype(np.float64).tobytes())
                            for text, vector in new_vecs.items()
                        ],
                    )

                con.commit()

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        """Compute embeddings for the given texts.

        Texts already in the index are returned from the cache.
        Uncached texts are computed on-the-fly but not persisted;
        use `index` to persist embeddings.
        """
        new_vecs = self._compute_vecs(texts)
        tmp_store = ChainMap(new_vecs, self.store)

        return [tmp_store[text] for text in texts]


@dataclass(slots=True, init=False)
class concat[V](BatchConversionFunc[V, NumpyArray]):
    """Concatenated embeddings of multiple models

    Args:
        models: List of embedding models
    """

    embed_funcs: Sequence[BatchConversionFunc[V, NumpyArray]]

    def __init__(
        self,
        funcs: Sequence[AnyConversionFunc[V, NumpyArray]],
    ):
        self.embed_funcs = [batchify_conversion(func) for func in funcs]

    def __call__(self, texts: Sequence[V]) -> Sequence[NumpyArray]:
        nested_vecs = [func(texts) for func in self.embed_funcs]
        return [np.concatenate(vecs, axis=0) for vecs in zip(*nested_vecs)]


__all__ = ["build", "cache", "concat", "embed_pairs"]
