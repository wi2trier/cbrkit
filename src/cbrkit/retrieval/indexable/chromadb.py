"""ChromaDB retriever wrapper."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import chromadb as cdb
from chromadb.api.types import SearchResult

from ...helpers import dist2sim
from ...indexable import chromadb as chromadb_storage
from ...typing import Casebase, SimMap
from ._common import SearchType, StorageRetriever, _normalize_results


@dataclass(slots=True)
class chromadb[K: str](StorageRetriever[K, chromadb_storage[K, Any]]):
    """Retriever wrapper for a ChromaDB storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.chromadb` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.  The retriever is a pure query path: index
    maintenance lives on the storage that owns the index (call
    ``storage.put_index(...)`` etc., or pass the storage to
    :func:`cbrkit.retain.indexable`).

    Extends :class:`~cbrkit.retrieval.indexable._common.StorageRetriever`
    for the batching/index-routing skeleton; ChromaDB issues one batched
    ``search`` for all queries (rather than the per-type leaves of
    :class:`VectorStorageRetriever`), so it implements :meth:`_search_db`
    directly and has no brute-force path.  Uses ChromaDB's `Search` API
    with `Knn` for dense/sparse ranking and `Rrf` for hybrid search.

    Args:
        storage: ChromaDB storage instance.
        search_type: `"dense"` (ANN), `"sparse"` (BM25/SPLADE),
            or `"hybrid"` (dense + sparse combined via RRF).
        limit: Max results per query (`None` = all).
        normalize_scores: Apply per-query min-max normalization.

    RRF parameters (``rrf_k``, ``rrf_weights``) are inherited from
    :class:`RrfMixin`.
    """

    search_type: SearchType = "dense"

    @override
    def _search_brute(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        raise NotImplementedError(
            "chromadb does not support brute-force search. Call put_index() first."
        )

    # -- search helpers ----------------------------------------------------

    def _process_result(
        self,
        result: SearchResult,
        score_transform: Callable[[float], float] = dist2sim,
    ) -> list[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Convert a ChromaDB SearchResult to result tuples."""
        output: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for query_rows in result.rows():
            cb: dict[K, str] = {}
            score_map: dict[K, float] = {}

            for row in query_rows:
                doc_id = row.get("id")

                if doc_id is None:
                    continue

                key = cast(K, doc_id)
                cb[key] = row.get("document") or ""
                score = row.get("score")

                if score is not None:
                    score_map[key] = score_transform(score)

            output.append((cb, score_map))

        return output

    def _build_rank(
        self,
        query: str,
        n: int,
    ) -> cdb.Knn | cdb.Rrf:
        """Build the ranking expression for the configured search type."""
        match self.search_type:
            case "dense":
                return cdb.Knn(query=query, limit=n)
            case "sparse":
                return cdb.Knn(
                    query=query,
                    key=self.storage.sparse_key,
                    limit=n,
                )
            case "hybrid":
                default_rank = n * 10
                return cdb.Rrf(
                    ranks=[
                        cdb.Knn(
                            query=query,
                            return_rank=True,
                            limit=n,
                            default=default_rank,
                        ),
                        cdb.Knn(
                            query=query,
                            key=self.storage.sparse_key,
                            return_rank=True,
                            limit=n,
                            default=default_rank,
                        ),
                    ],
                    weights=list(self.rrf_weights),
                    k=self.rrf_k,
                )
            case _:
                raise ValueError(f"Unknown search_type: {self.search_type!r}")

    @override
    def _search_db(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage._collection is not None

        n = (
            self.limit
            if self.limit is not None
            else self.storage._collection.count()
        )

        if n == 0:
            return [({}, {}) for _ in queries]

        searches = [
            cdb.Search()
            .rank(self._build_rank(q, n))
            .limit(n)
            .select(cdb.K.DOCUMENT, cdb.K.SCORE)
            for q in queries
        ]

        score_transform: Callable[[float], float] = (
            (lambda s: -s) if self.search_type == "hybrid" else dist2sim
        )

        return _normalize_results(
            self._process_result(
                self.storage._collection.search(searches),
                score_transform=score_transform,
            ),
            self.normalize_scores,
        )


__all__ = ["chromadb"]
