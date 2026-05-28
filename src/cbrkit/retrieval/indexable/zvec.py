"""zvec retriever wrapper."""

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Literal, cast, override

import numpy as np
import zvec as zv  # pyright: ignore[reportMissingImports]  # type: ignore[unresolved-import]

from ...helpers import dispatch_batches, dist2sim
from ...indexable import zvec as zvec_storage
from ...typing import (
    BatchConversionFunc,
    Casebase,
    IndexableFunc,
    NumpyArray,
    RetrieverFunc,
    SimMap,
    SparseVector,
)
from ._common import (
    _StorageIndexMixin,
    _brute_force_dense_search,
    _normalize_results,
)


@dataclass(slots=True)
class zvec[K: str](
    _StorageIndexMixin[K, zvec_storage[K]],
    RetrieverFunc[K, str, float],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """Retriever wrapper for a zvec storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.zvec` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.

    Uses zvec's built-in `RrfReRanker` for hybrid search.

    Args:
        storage: Zvec storage instance.
        search_type: Search mode — `"dense"` (ANN), `"sparse"`
            (sparse vectors), or `"hybrid"` (dense + sparse
            combined via RRF).
        query_conversion_func: Optional separate dense embedding
            function for queries.  Falls back to the storage's
            `conversion_func`.
        sparse_query_conversion_func: Optional separate sparse
            embedding function for queries.  Falls back to the
            storage's `sparse_conversion_func`.
        limit: Maximum number of results to return per query.
            `None` (default) returns all indexed documents.
        filter: Optional filter expression applied during every
            search query.
        rrf_k: RRF smoothing parameter for hybrid search
            (default: 60).
        normalize_scores: Apply per-query min-max normalization.
    """

    storage: zvec_storage[K]
    search_type: Literal["dense", "sparse", "hybrid"] = "dense"
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    sparse_query_conversion_func: BatchConversionFunc[str, SparseVector] | None = (
        None
    )
    limit: int | None = None
    filter: str | None = None
    rrf_k: int = 60
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

        return self._search_brute(queries, casebase)

    # -- search helpers ----------------------------------------------------

    def _search_limit(self) -> int:
        """Return the effective search limit."""
        n = self.storage.search_limit()

        if n is None:
            return self.limit or 10

        if self.limit is not None:
            return min(self.limit, n)

        return n

    def _docs_to_result(
        self,
        docs: list[zv.Doc],
        score_transform: Callable[[float], float] = dist2sim,
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        """Convert zvec Doc list to a (casebase, score_map) pair."""
        cb: dict[K, str] = {}
        sm: dict[K, float] = {}

        for doc in docs:
            key = cast(K, doc.id)
            cb[key] = cast(
                str,
                doc.field(self.storage.value_field)
                if doc.has_field(self.storage.value_field)
                else "",
            )
            if doc.score is not None:
                sm[key] = score_transform(doc.score)

        return cb, sm

    def _search_brute(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Brute-force dense vector search for non-indexed casebases."""
        if self.search_type != "dense":
            raise NotImplementedError(
                f"Brute-force search is not supported for search_type={self.search_type!r}. "
                "Call put_index() first."
            )

        assert self.storage.conversion_func is not None
        return _brute_force_dense_search(
            queries,
            casebase,
            self.storage.conversion_func,
            self.query_conversion_func,
        )

    def _search_db(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Dispatch to the appropriate search method and normalize scores."""
        match self.search_type:
            case "dense":
                results = self._search_db_dense(queries)
            case "sparse":
                results = self._search_db_sparse(queries)
            case "hybrid":
                results = self._search_db_hybrid(queries)

        return _normalize_results(results, self.normalize_scores)

    def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage._collection is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._search_limit()

        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for qvec in query_vecs:
            docs = self.storage._collection.query(
                zv.VectorQuery(
                    self.storage.dense_vector_name,
                    vector=np.asarray(qvec).tolist(),
                ),
                topk=n,
                filter=self.filter,
                output_fields=[self.storage.value_field],
            )
            cb, sm = self._docs_to_result(docs)
            results.append((cb, sm))

        return results

    def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage._collection is not None
        assert self.storage.sparse_conversion_func is not None

        sparse_func = (
            self.sparse_query_conversion_func or self.storage.sparse_conversion_func
        )
        query_vecs = sparse_func(list(queries))
        n = self._search_limit()

        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for svec in query_vecs:
            docs = self.storage._collection.query(
                zv.VectorQuery(
                    self.storage.sparse_vector_name,
                    vector=svec,
                ),
                topk=n,
                filter=self.filter,
                output_fields=[self.storage.value_field],
            )

            if not docs:
                results.append(({}, {}))
                continue

            cb, sm = self._docs_to_result(docs)
            results.append((cb, sm))

        return results

    def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage._collection is not None
        assert self.storage.conversion_func is not None
        assert self.storage.sparse_conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        sparse_func = (
            self.sparse_query_conversion_func or self.storage.sparse_conversion_func
        )

        query_vecs = embed_func(list(queries))
        sparse_query_vecs = sparse_func(list(queries))
        n = self._search_limit()

        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for qvec, svec in zip(query_vecs, sparse_query_vecs, strict=True):
            docs = self.storage._collection.query(
                [
                    zv.VectorQuery(
                        self.storage.dense_vector_name,
                        vector=np.asarray(qvec).tolist(),
                    ),
                    zv.VectorQuery(
                        self.storage.sparse_vector_name,
                        vector=svec,
                    ),
                ],
                topk=n,
                filter=self.filter,
                output_fields=[self.storage.value_field],
                reranker=zv.RrfReRanker(rank_constant=self.rrf_k),
            )

            if not docs:
                results.append(({}, {}))
                continue

            # RRF scores are relevance-ordered (higher = better)
            cb, sm = self._docs_to_result(docs, score_transform=lambda s: s)
            results.append((cb, sm))

        return results


__all__ = ["zvec"]
