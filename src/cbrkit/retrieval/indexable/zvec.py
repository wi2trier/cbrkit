"""zvec retriever wrapper."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast, override

import numpy as np
import zvec as zv  # pyright: ignore[reportMissingImports]  # type: ignore[unresolved-import]

from ...helpers import dist2sim
from ...indexable import zvec as zvec_storage
from ...typing import (
    BatchConversionFunc,
    Casebase,
    SimMap,
    SparseVector,
)
from ._common import VectorStorageRetriever


@dataclass(slots=True)
class zvec[K: str](VectorStorageRetriever[K, zvec_storage[K, Any]]):
    """Retriever wrapper for a zvec storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.zvec` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.  The retriever is a pure query path: index
    maintenance lives on the storage that owns the index (call
    ``storage.put_index(...)`` etc., or pass the storage to
    :func:`cbrkit.retain.indexable`).

    The dispatch skeleton (batching, index/brute routing, search-type
    dispatch, normalization) is inherited from
    :class:`~cbrkit.retrieval.indexable._common.VectorStorageRetriever`;
    only the three ``_search_db_*`` leaves are implemented here.  Uses
    zvec's built-in `RrfReRanker` for hybrid search.

    Note:
        When a call carries multiple queries they are executed
        sequentially: zvec's embedded client takes one query at a time,
        so there is no concurrent fan-out to exploit.

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

    sparse_query_conversion_func: BatchConversionFunc[str, SparseVector] | None = (
        None
    )
    filter: str | None = None

    # -- search helpers ----------------------------------------------------

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

    @override
    def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        assert self.storage._collection is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._clamp_search_limit() or 10

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

    @override
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
        n = self._clamp_search_limit() or 10

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

    @override
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
        n = self._clamp_search_limit() or 10

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
