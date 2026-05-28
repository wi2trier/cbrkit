"""ChromaDB retriever wrapper."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Literal, cast, override

import chromadb as cdb
from chromadb.api.types import SearchResult

from ...helpers import dispatch_batches, dist2sim
from ...indexable import chromadb as chromadb_storage
from ...typing import Casebase, IndexableFunc, RetrieverFunc, SimMap
from ._common import RrfMixin, _StorageIndexMixin, _normalize_results


@dataclass(slots=True)
class chromadb[K: str](
    _StorageIndexMixin[K, chromadb_storage[K]],
    RrfMixin,
    RetrieverFunc[K, str, float],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """Retriever wrapper for a ChromaDB storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.chromadb` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.

    Uses ChromaDB's `Search` API with `Knn` for dense/sparse
    ranking and `Rrf` (Reciprocal Rank Fusion) for hybrid search.

    Args:
        storage: ChromaDB storage instance.
        search_type: `"dense"` (ANN), `"sparse"` (BM25/SPLADE),
            or `"hybrid"` (dense + sparse combined via RRF).
        limit: Max results per query (`None` = all).
        normalize_scores: Apply per-query min-max normalization.

    RRF parameters (``rrf_k``, ``rrf_weights``) are inherited from
    :class:`RrfMixin`.
    """

    storage: chromadb_storage[K]
    search_type: Literal["dense", "sparse", "hybrid"] = "dense"
    limit: int | None = None
    normalize_scores: bool = True

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if not batches:
            return []

        return dispatch_batches(batches, self._dispatch)

    def _dispatch(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        if len(casebase) == 0:
            if not self.storage.has_index():
                raise ValueError(
                    "Indexed retrieval was requested with an empty casebase, "
                    "but no index is available. Call put_index() first."
                )

            return self._search_db(queries)

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
