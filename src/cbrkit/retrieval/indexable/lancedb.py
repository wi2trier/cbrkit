"""LanceDB retriever wrapper."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Literal, override

import numpy as np

from ...helpers import dispatch_batches, dist2sim
from ...indexable import lancedb as lancedb_storage
from ...typing import (
    BatchConversionFunc,
    Casebase,
    IndexableFunc,
    NumpyArray,
    RetrieverFunc,
    SimMap,
)
from ._common import (
    _StorageIndexMixin,
    _brute_force_dense_search,
    _normalize_results,
)


@dataclass(slots=True)
class lancedb[K: int | str](
    _StorageIndexMixin[K, lancedb_storage[K]],
    RetrieverFunc[K, str, float],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """Retriever wrapper for a LanceDB storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.lancedb` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.

    Args:
        storage: LanceDB storage instance.
        search_type: Search mode — `"dense"` (ANN), `"sparse"`
            (BM25), or `"hybrid"` (dense + sparse combined).
        query_conversion_func: Optional separate embedding function
            for queries.  Falls back to the storage's
            `conversion_func`.
        limit: Maximum number of results to return per query.
            `None` (default) returns all rows.
        where: Optional SQL filter expression applied during every
            search query (e.g. `"category = 'A'"`).
        normalize_scores: Apply per-query min-max normalization.
    """

    storage: lancedb_storage[K]
    search_type: Literal["dense", "sparse", "hybrid"] = "dense"
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    limit: int | None = None
    where: str | None = None
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

    def _search_limit(self) -> int | None:
        """Return the effective search limit."""
        n = self.storage.search_limit()

        if n is None:
            return self.limit

        if self.limit is not None:
            return min(self.limit, n)

        return n

    def _hits_to_result(
        self,
        hits: list[dict[str, Any]],
        score_key: str,
    ) -> tuple[Casebase[K, str], SimMap[K, float]]:
        """Convert LanceDB hit dicts to a (casebase, score_map) pair."""
        kc = self.storage.key_column
        vc = self.storage.value_column
        return (
            {hit[kc]: hit[vc] for hit in hits},
            {hit[kc]: float(hit[score_key]) for hit in hits},
        )

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
        table = self.storage._table
        assert table is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._search_limit()

        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for qvec in query_vecs:
            q = table.search(
                np.asarray(qvec).tolist(),
                vector_column_name=self.storage.vector_column,
            ).limit(n)

            if self.where is not None:
                q = q.where(self.where)

            hits = q.to_list()
            cb, sm = self._hits_to_result(hits, "_distance")
            results.append((cb, {k: dist2sim(v) for k, v in sm.items()}))

        return results

    def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        table = self.storage._table
        assert table is not None

        n = self._search_limit()
        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for query in queries:
            q = table.search(query, query_type="fts").limit(n)

            if self.where is not None:
                q = q.where(self.where)

            hits = q.to_list()

            if not hits:
                results.append(({}, {}))
                continue

            results.append(self._hits_to_result(hits, "_score"))

        return results

    def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        table = self.storage._table
        assert table is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._search_limit()

        results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

        for query, qvec in zip(queries, query_vecs, strict=True):
            q = (
                table.search(query_type="hybrid")
                .vector(np.asarray(qvec).tolist())
                .text(query)
                .limit(n)
            )

            if self.where is not None:
                q = q.where(self.where)

            hits = q.to_list()

            if not hits:
                results.append(({}, {}))
                continue

            results.append(self._hits_to_result(hits, "_relevance_score"))

        return results


__all__ = ["lancedb"]
