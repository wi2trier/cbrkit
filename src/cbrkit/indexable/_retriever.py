"""Shared skeletons of the retrievers that search a storage backend."""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, override

from .._phase import Ranking
from .._util import RRF_K, fuse_rankings
from ..sim.embed import for_queries
from ..typing import AsyncCbrFunc, Batches, EmbeddingFunc, NumpyArray, Results

type IndexType = Literal["dense", "sparse", "hybrid"]
"""What a storage indexes, embeddings, full text, or both, which decides the searches it supports."""


@dataclass(slots=True, frozen=True)
class Dense:
    """Nearest neighbors of the query embedding by the vector metric of the storage."""


@dataclass(slots=True, frozen=True)
class Sparse:
    """Full-text search of the query, ranked by the backend, such as BM25."""


@dataclass(slots=True, frozen=True)
class Hybrid:
    """Dense and sparse search fused by weighted reciprocal rank fusion.

    Args:
        k: Smoothing constant of the fusion, which scores a rank by `1 / (k + rank)`.
        weights: Weights of the dense and the sparse ranking.
    """

    k: int = RRF_K
    weights: tuple[float, float] = (0.7, 0.3)


type Search = Dense | Sparse | Hybrid
"""How a storage retriever searches, a dense, sparse or hybrid search."""


def default_search(index_type: IndexType) -> Search:
    """The search that uses everything a storage indexes, which its retrievers run if given no search.

    Args:
        index_type: What the storage indexes.

    Examples:
        >>> default_search("hybrid")
        Hybrid(k=60, weights=(0.7, 0.3))
    """
    match index_type:
        case "dense":
            return Dense()
        case "sparse":
            return Sparse()
        case "hybrid":
            return Hybrid()


def has_dense(index_type: IndexType) -> bool:
    """Whether a storage of this index type keeps a dense vector index.

    Examples:
        >>> has_dense("hybrid"), has_dense("sparse")
        (True, False)
    """
    return index_type in ("dense", "hybrid")


def has_sparse(index_type: IndexType) -> bool:
    """Whether a storage of this index type keeps a sparse full-text index.

    Examples:
        >>> has_sparse("hybrid"), has_sparse("dense")
        (True, False)
    """
    return index_type in ("sparse", "hybrid")


def check_embedder(
    index_type: IndexType, embedder: EmbeddingFunc[str, NumpyArray] | None
) -> None:
    """Raises a `ValueError` if a dense index has no embedder."""
    if has_dense(index_type) and embedder is None:
        raise ValueError(f"embedder is required for index_type={index_type!r}")


def dense_embedder(
    index_type: IndexType, embedder: EmbeddingFunc[str, NumpyArray] | None
) -> EmbeddingFunc[str, NumpyArray] | None:
    """The embedder of a storage that keeps a dense vector index, None for one that keeps none.

    Examples:
        >>> dense_embedder("sparse", print) is None, dense_embedder("hybrid", print) is print
        (True, True)
    """
    return embedder if has_dense(index_type) else None


async def embed_rows(
    embedder: EmbeddingFunc[str, NumpyArray],
    rows: Sequence[dict[str, Any]],
    text_column: str,
    vector_key: str,
    encode: Callable[[NumpyArray], Any],
) -> None:
    """Stores the encoded embedding of the text of every row under `vector_key`."""
    texts = [row[text_column] for row in rows]

    for row, vec in zip(rows, await embedder(texts), strict=True):
        row[vector_key] = encode(vec)


class _VectorStorage(Protocol):
    """Storage that a retriever searches."""

    @property
    def embedder(self) -> EmbeddingFunc[str, NumpyArray] | None: ...

    @property
    def index_type(self) -> IndexType: ...


@dataclass(slots=True, frozen=True, kw_only=True)
class VectorStorageRetriever[K, St: _VectorStorage](
    Ranking, AsyncCbrFunc[K, str, float], ABC
):
    """Skeleton of the retrievers of storages with dense, sparse and hybrid search.

    It searches its storage and ignores the cases it is given,
    so it comes first and `cbrkit.cycle.run` gets None as the source.
    Subclasses provide the leaves `_search_dense` and `_search_sparse`,
    whose rankings `_search_hybrid` fuses unless a backend searches both at once.
    It keeps the best cases by the fields of `cbrkit.retrieve.Ranking`, normalized by default.

    Args:
        storage: The searched storage.
        search: How the storage is searched, everything it indexes if None.
        query_embedder: Embeds the queries, the `embedder` of the storage if None.
        normalize_scores: Scale the scores of every query to between 0 and 1.
    """

    storage: St
    search: Search | None = None
    query_embedder: EmbeddingFunc[str, NumpyArray] | None = None
    normalize_scores: bool = True

    @override
    async def __call__(self, batches: Batches[K, str], /) -> Results[K, str]:
        queries = [query for _, query in batches]

        if not queries:
            return [({}, {}) for _ in queries]

        match self.search or default_search(self.storage.index_type):
            case Dense():
                results = await self._search_dense(queries)
            case Sparse():
                results = await self._search_sparse(queries)
            case Hybrid() as search:
                results = await self._search_hybrid(queries, search)

        return self.select(results)

    async def _embed(self, queries: Sequence[str]) -> Sequence[NumpyArray]:
        """Embed the queries with `query_embedder` or the `embedder` of the storage."""
        embedder = self.query_embedder or self.storage.embedder

        if embedder is None:
            raise ValueError("A dense search needs an embedder")

        return await for_queries(embedder)(queries)

    @abstractmethod
    async def _search_dense(self, queries: Sequence[str]) -> Results[K, str]: ...

    @abstractmethod
    async def _search_sparse(self, queries: Sequence[str]) -> Results[K, str]: ...

    async def _search_hybrid(
        self, queries: Sequence[str], search: Hybrid
    ) -> Results[K, str]:
        dense, sparse = await asyncio.gather(
            self._search_dense(queries), self._search_sparse(queries)
        )

        return [
            fuse_rankings(rankings, search.weights, search.k, self.limit)
            for rankings in zip(dense, sparse, strict=True)
        ]


__all__ = [
    "Dense",
    "Hybrid",
    "IndexType",
    "Search",
    "Sparse",
    "VectorStorageRetriever",
    "check_embedder",
    "default_search",
    "dense_embedder",
    "embed_rows",
    "has_dense",
    "has_sparse",
]
