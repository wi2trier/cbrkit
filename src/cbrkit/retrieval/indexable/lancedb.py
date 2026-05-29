"""LanceDB retriever wrapper."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, override

import numpy as np

from ...helpers import dist2sim
from ...indexable import lancedb as lancedb_storage
from ...typing import Casebase, SimMap
from ._common import VectorStorageRetriever


@dataclass(slots=True)
class lancedb[K: int | str](VectorStorageRetriever[K, lancedb_storage[K, Any]]):
    """Retriever wrapper for a LanceDB storage backend.

    Delegates storage to a :class:`~cbrkit.indexable.lancedb` instance
    and performs search queries against it.  Multiple retriever
    instances can share the same storage to query with different
    search types.  The retriever is a pure query path: index
    maintenance lives on the storage that owns the index (call
    ``storage.put_index(...)`` etc., or pass the storage to
    :func:`cbrkit.retain.indexable`).

    The dispatch skeleton (batching, index/brute routing, search-type
    dispatch, normalization) is inherited from
    :class:`~cbrkit.retrieval.indexable._common.VectorStorageRetriever`;
    only the three ``_search_db_*`` leaves are implemented here.

    Note:
        When a call carries multiple queries they are executed
        sequentially: LanceDB's embedded client takes one query at a
        time, so there is no concurrent fan-out to exploit.

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

    where: str | None = None

    # -- search helpers ----------------------------------------------------

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

    @override
    def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        table = self.storage._table
        assert table is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._clamp_search_limit()

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

    @override
    def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        table = self.storage._table
        assert table is not None

        n = self._clamp_search_limit()
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

    @override
    def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        table = self.storage._table
        assert table is not None
        assert self.storage.conversion_func is not None

        embed_func = self.query_conversion_func or self.storage.conversion_func
        query_vecs = embed_func(list(queries))
        n = self._clamp_search_limit()

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
