from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from ..helpers import (
    batchify_sim,
    dispatch_batches,
    dist2sim,
    get_logger,
    normalize,
    optional_dependencies,
)
from ..indexable import _compute_index_diff
from ..sim.embed import cache, default_score_func
from ..typing import (
    AnySimFunc,
    BatchConversionFunc,
    BatchSimFunc,
    Casebase,
    Float,
    IndexableFunc,
    NumpyArray,
    RetrieverFunc,
    SimMap,
    SparseVector,
)

logger = get_logger(__name__)


def resolve_casebases[K, V](
    batches: Sequence[tuple[Casebase[K, V], V]],
    indexed_casebase: Casebase[K, V] | None,
) -> list[tuple[Casebase[K, V], V]]:
    """Resolve casebases for indexable retrievers.

    Empty casebases are treated as an explicit signal to use indexed retrieval mode.
    In indexed mode, empty casebases are replaced with the previously indexed casebase.
    """
    if indexed_casebase is None:
        if any(len(casebase) == 0 for casebase, _ in batches):
            raise ValueError(
                "Indexed retrieval was requested with an empty casebase, but no index is available. "
                "Call create_index() first."
            )

        return list(batches)

    return [
        (indexed_casebase if len(casebase) == 0 else casebase, query)
        for casebase, query in batches
    ]


def _normalize_results[K](
    results: Sequence[tuple[Casebase[K, str], SimMap[K, float]]],
    enabled: bool,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Apply per-query min-max normalization if enabled."""
    if not enabled:
        return results

    normalized: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for cb, sm in results:
        if not sm:
            normalized.append((cb, sm))
            continue

        mn = min(sm.values())
        mx = max(sm.values())
        normalized.append((cb, {k: normalize(v, mn, mx) for k, v in sm.items()}))

    return normalized


def _brute_force_dense_search[K](
    queries: Sequence[str],
    casebase: Casebase[K, str],
    conversion_func: BatchConversionFunc[str, NumpyArray],
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Shared brute-force dense vector search for non-indexed casebases."""
    keys = list(casebase.keys())
    values = list(casebase.values())

    case_vecs = conversion_func(values)
    embed_func = query_conversion_func or conversion_func
    query_vecs = embed_func(list(queries))

    sim_func = batchify_sim(default_score_func)

    results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for qvec in query_vecs:
        sims = sim_func([(cv, qvec) for cv in case_vecs])
        results.append(
            (
                dict(casebase),
                dict(zip(keys, sims, strict=True)),
            )
        )

    return results


@dataclass(slots=True, init=False)
class embed[K, S: Float](
    RetrieverFunc[K, str, S],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """Embedding-based semantic retriever with indexing support.

    Args:
        conversion_func: Embedding function (from embed module).
        sim_func: Vector similarity function (default: cosine).
        query_conversion_func: Optional separate embedding function for queries.

    Pass an empty casebase to ``__call__`` to use the pre-indexed casebase.
    """

    conversion_func: cache
    sim_func: BatchSimFunc[NumpyArray, S]
    query_conversion_func: cache | None
    _casebase: dict[K, str] | None = field(repr=False, init=False, default=None)

    def __init__(
        self,
        conversion_func: cache,
        sim_func: AnySimFunc[NumpyArray, S] = default_score_func,  # type: ignore[assignment]
        query_conversion_func: cache | None = None,
    ):
        self.conversion_func = conversion_func
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = query_conversion_func
        self._casebase = None

    @override
    def has_index(self) -> bool:
        """Return whether an embedding index has been created."""
        return self._casebase is not None

    @property
    @override
    def index(self) -> Casebase[K, str]:
        """Return the indexed casebase."""
        if self._casebase is None:
            return {}
        return self._casebase

    @override
    def create_index(self, data: Casebase[K, str]) -> None:
        """Ensure the embedding index exists and sync it with *data*.

        On first call the casebase and embedding cache are built from
        scratch.  On subsequent calls the existing casebase is diffed
        against *data* and only stale or changed entries are
        deleted/added via :meth:`delete_index` and :meth:`update_index`.
        """
        if self._casebase is None:
            self._casebase = dict(data)
            self.conversion_func.create_index(data.values())
            return

        existing = self._casebase
        stale_keys, changed_or_new = _compute_index_diff(existing, data)

        if not stale_keys and not changed_or_new:
            return

        if stale_keys:
            self.delete_index(stale_keys)

        if changed_or_new:
            self.update_index(changed_or_new)

    @override
    def update_index(self, data: Casebase[K, str]) -> None:
        """Add new entries to the embedding index.

        If no index exists yet, delegates to :meth:`create_index`.
        """
        if self._casebase is None:
            self.create_index(data)
            return

        if not data:
            return

        self.conversion_func.update_index(data.values())
        self._casebase.update(data)

    @override
    def delete_index(self, data: Collection[K]) -> None:
        """Remove entries by key from the embedding index."""
        if self._casebase is None or not data:
            return

        texts_to_delete = [self._casebase[key] for key in data if key in self._casebase]

        if texts_to_delete:
            self.conversion_func.delete_index(texts_to_delete)

        for key in data:
            self._casebase.pop(key, None)

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], dict[K, S]]]:
        if not batches:
            return []

        resolved = resolve_casebases(batches, self._casebase)
        sim_maps = dispatch_batches(resolved, self.__call_queries__)

        return [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(resolved, sim_maps, strict=True)
        ]

    def __call_queries__(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[dict[K, S]]:
        case_texts = list(casebase.values())
        query_texts = list(queries)

        if self.query_conversion_func:
            case_vecs = self.conversion_func(case_texts)
            query_vecs = self.query_conversion_func(query_texts)
        else:
            all_texts = case_texts + query_texts
            all_vecs = self.conversion_func(all_texts)
            case_vecs = all_vecs[: len(case_texts)]
            query_vecs = all_vecs[len(case_texts) :]

        case_keys = list(casebase.keys())

        return [
            dict(
                zip(
                    case_keys,
                    self.sim_func([(cv, query_vec) for cv in case_vecs]),
                    strict=True,
                )
            )
            for query_vec in query_vecs
        ]


with optional_dependencies():
    import bm25s
    import numpy as np

    from ..sim.embed import bm25 as bm25_embed

    @dataclass(slots=True)
    class bm25[K](
        RetrieverFunc[K, str, float],
        IndexableFunc[Casebase[K, str], Collection[K]],
    ):
        """BM25 retriever based on bm25s.

        Delegates BM25 model management to a
        :class:`~cbrkit.sim.embed.bm25` instance and performs
        BM25 scoring for retrieval.

        Args:
            conversion_func: BM25 sparse embedding function
                (from :mod:`cbrkit.sim.embed`).
            normalize_scores: If ``True`` (default), apply per-query min-max
                normalization to BM25 scores. If ``False``, return raw BM25
                scores.

        Pass an empty casebase to ``__call__`` to use the pre-indexed casebase.
        """

        conversion_func: bm25_embed
        normalize_scores: bool = True
        _keys: list[K] | None = field(default=None, init=False, repr=False)

        @override
        def has_index(self) -> bool:
            """Return whether a BM25 index has been created."""
            return self._keys is not None

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase."""
            corpus = self.conversion_func._corpus
            if self._keys is None or corpus is None:
                return {}
            return dict(zip(self._keys, corpus, strict=True))

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild BM25 index."""
            self._keys = list(data.keys())
            self.conversion_func.create_index(data.values())

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Merge new data with existing casebase and rebuild index."""
            if self._keys is None:
                self.create_index(data)
                return

            merged = dict(self.index)
            merged.update(data)
            self.create_index(merged)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Remove keys from the casebase and rebuild index."""
            if self._keys is None:
                return

            remove = set(data)
            remaining = {k: v for k, v in self.index.items() if k not in remove}
            self.create_index(remaining)

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
            indexed = self.index or None
            resolved = resolve_casebases(batches, indexed)

            def call_queries(
                queries: Sequence[str],
                casebase: Casebase[K, str],
            ) -> Sequence[dict[K, float]]:
                return self.__call_queries__(queries, casebase, indexed)

            sim_maps = dispatch_batches(resolved, call_queries)

            return [
                (casebase, sim_map)
                for (casebase, _), sim_map in zip(resolved, sim_maps, strict=True)
            ]

        def __call_queries__(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
            indexed: Casebase[K, str] | None,
        ) -> Sequence[dict[K, float]]:
            if (
                self.conversion_func._retriever is not None
                and indexed is not None
                and casebase is indexed
            ):
                retriever = self.conversion_func._retriever
            else:
                retriever = self.conversion_func._build_retriever(casebase.values())

            queries_tokens = bm25s.tokenize(
                cast(list[str], queries),
                stemmer=self.conversion_func._stemmer,
                stopwords=self.conversion_func._stopwords,
            )

            results, scores = retriever.retrieve(
                queries_tokens,
                sorted=False,
                k=len(casebase),
            )

            key_index = {idx: key for idx, key in enumerate(casebase)}

            if self.normalize_scores:
                normalized_scores: list[list[float]] = []

                for query_scores in scores:
                    query_min = float(np.min(query_scores))
                    query_max = float(np.max(query_scores))
                    normalized_scores.append(
                        [
                            normalize(float(score), query_min, query_max)
                            for score in query_scores
                        ]
                    )

                return [
                    {
                        key_index[case_id]: score
                        for case_id, score in zip(
                            results[query_id], normalized_scores[query_id], strict=True
                        )
                    }
                    for query_id in range(len(queries))
                ]

            return [
                {
                    key_index[case_id]: float(score)
                    for case_id, score in zip(
                        results[query_id], scores[query_id], strict=True
                    )
                }
                for query_id in range(len(queries))
            ]


with optional_dependencies():
    import numpy as np

    from ..indexable import lancedb as lancedb_storage

    @dataclass(slots=True)
    class lancedb[K: int | str](
        RetrieverFunc[K, str, float],
    ):
        """Retriever wrapper for a LanceDB storage backend.

        Delegates storage to a :class:`~cbrkit.indexable.lancedb` instance
        and performs search queries against it.  Multiple retriever
        instances can share the same storage to query with different
        search types.

        Args:
            storage: LanceDB storage instance.
            search_type: Search mode — ``"dense"`` (ANN), ``"sparse"``
                (BM25), or ``"hybrid"`` (dense + sparse combined).
            query_conversion_func: Optional separate embedding function
                for queries.  Falls back to the storage's
                ``conversion_func``.
            limit: Maximum number of results to return per query.
                ``None`` (default) returns all rows.
            where: Optional SQL filter expression applied during every
                search query (e.g. ``"category = 'A'"``).
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
                        "but no index is available. Call create_index() first."
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
                    "Call create_index() first."
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
            assert self.storage._table is not None
            assert self.storage.conversion_func is not None

            embed_func = self.query_conversion_func or self.storage.conversion_func
            query_vecs = embed_func(list(queries))
            n = self._search_limit()

            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for qvec in query_vecs:
                q = self.storage._table.search(
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
            assert self.storage._table is not None

            n = self._search_limit()
            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for query in queries:
                q = self.storage._table.search(query, query_type="fts").limit(n)

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
            assert self.storage._table is not None
            assert self.storage.conversion_func is not None

            embed_func = self.query_conversion_func or self.storage.conversion_func
            query_vecs = embed_func(list(queries))
            n = self._search_limit()

            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for query, qvec in zip(queries, query_vecs, strict=True):
                q = (
                    self.storage._table.search(query_type="hybrid")
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


with optional_dependencies():
    import chromadb as cdb
    from chromadb.api.types import SearchResult

    from ..indexable import chromadb as chromadb_storage

    @dataclass(slots=True)
    class chromadb[K: str](
        RetrieverFunc[K, str, float],
    ):
        """Retriever wrapper for a ChromaDB storage backend.

        Delegates storage to a :class:`~cbrkit.indexable.chromadb` instance
        and performs search queries against it.  Multiple retriever
        instances can share the same storage to query with different
        search types.

        Uses ChromaDB's ``Search`` API with ``Knn`` for dense/sparse
        ranking and ``Rrf`` (Reciprocal Rank Fusion) for hybrid search.

        Args:
            storage: ChromaDB storage instance.
            search_type: ``"dense"`` (ANN), ``"sparse"`` (BM25/SPLADE),
                or ``"hybrid"`` (dense + sparse combined via RRF).
            limit: Max results per query (``None`` = all).
            rrf_k: Smoothing parameter for RRF (default: 60).
            rrf_weights: Weights for ``(dense, sparse)`` in RRF
                (default: ``(0.7, 0.3)``).
            normalize_scores: Apply per-query min-max normalization.
        """

        storage: chromadb_storage[K]
        search_type: Literal["dense", "sparse", "hybrid"] = "dense"
        limit: int | None = None
        rrf_k: int = 60
        rrf_weights: tuple[float, float] = field(default=(0.7, 0.3))
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
                        "but no index is available. Call create_index() first."
                    )

                return self._search_db(queries)

            raise NotImplementedError(
                "chromadb does not support brute-force search. "
                "Call create_index() first."
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


with optional_dependencies():
    import numpy as np
    import zvec as zv

    from ..indexable import zvec as zvec_storage

    @dataclass(slots=True)
    class zvec[K: str](
        RetrieverFunc[K, str, float],
    ):
        """Retriever wrapper for a zvec storage backend.

        Delegates storage to a :class:`~cbrkit.indexable.zvec` instance
        and performs search queries against it.  Multiple retriever
        instances can share the same storage to query with different
        search types.

        Uses zvec's built-in ``RrfReRanker`` for hybrid search.

        Args:
            storage: Zvec storage instance.
            search_type: Search mode — ``"dense"`` (ANN), ``"sparse"``
                (sparse vectors), or ``"hybrid"`` (dense + sparse
                combined via RRF).
            query_conversion_func: Optional separate dense embedding
                function for queries.  Falls back to the storage's
                ``conversion_func``.
            sparse_query_conversion_func: Optional separate sparse
                embedding function for queries.  Falls back to the
                storage's ``sparse_conversion_func``.
            limit: Maximum number of results to return per query.
                ``None`` (default) returns all indexed documents.
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
                        "but no index is available. Call create_index() first."
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
                    "Call create_index() first."
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
