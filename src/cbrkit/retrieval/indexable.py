from abc import ABC, abstractmethod
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


class _DatabaseRetriever[K](
    RetrieverFunc[K, str, float],
    IndexableFunc[Casebase[K, str], Collection[K]],
    ABC,
):
    """Base class for database-backed retrievers.

    Subclasses are dataclasses that provide fields and override abstract
    methods.  This class supplies ``__call__``, dispatch logic, and
    brute-force dense vector search.
    """

    # Annotations only — actual fields provided by dataclass subclasses.
    search_type: Literal["dense", "sparse", "hybrid"]
    normalize_scores: bool

    # -- concrete helpers --------------------------------------------------

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
            if not self._has_index():
                raise ValueError(
                    "Indexed retrieval was requested with an empty casebase, "
                    "but no index is available. Call create_index() first."
                )

            return self._search_db(queries)

        return self._search_brute(queries, casebase)

    def _search_brute(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Brute-force search fallback for non-indexed casebases.

        Subclasses that support brute-force search should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support brute-force search. "
            "Call create_index() first."
        )

    # -- DB search dispatch + normalization --------------------------------

    def _normalize_results(
        self,
        results: Sequence[tuple[Casebase[K, str], SimMap[K, float]]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        """Apply per-query min-max normalization if enabled."""
        if not self.normalize_scores:
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

        return self._normalize_results(results)

    def _search_db_dense(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support dense search"
        )

    def _search_db_sparse(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support sparse search"
        )

    def _search_db_hybrid(
        self,
        queries: Sequence[str],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        raise NotImplementedError(
            f"{type(self).__name__} does not support hybrid search"
        )

    # -- abstract interface ------------------------------------------------

    @abstractmethod
    def _has_index(self) -> bool: ...

    @property
    @abstractmethod
    def index(self) -> Casebase[K, str]: ...

    @abstractmethod
    def create_index(self, data: Casebase[K, str]) -> None: ...

    @abstractmethod
    def update_index(self, data: Casebase[K, str]) -> None: ...

    @abstractmethod
    def delete_index(self, data: Collection[K]) -> None: ...


@dataclass(slots=True, init=False)
class embed[K, S: Float](
    RetrieverFunc[K, str, S], IndexableFunc[Casebase[K, str], Collection[K]]
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
        sim_func: AnySimFunc[NumpyArray, S] = default_score_func,
        query_conversion_func: cache | None = None,
    ):
        self.conversion_func = conversion_func
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = query_conversion_func
        self._casebase = None

    @property
    @override
    def index(self) -> Casebase[K, str]:
        """Return the indexed casebase."""
        if self._casebase is None:
            return {}
        return self._casebase

    @override
    def create_index(self, data: Casebase[K, str]) -> None:
        """Rebuild embedding index, reusing cached embeddings where possible."""
        self._casebase = dict(data)
        self.conversion_func.create_index(data.values())

    @override
    def update_index(self, data: Casebase[K, str]) -> None:
        """Add new entries to an existing index."""
        if self._casebase is None:
            self.create_index(data)
            return

        self._casebase.update(data)
        self.conversion_func.update_index(data.values())

    @override
    def delete_index(self, data: Collection[K]) -> None:
        """Remove entries by key."""
        if self._casebase is None:
            return

        texts_to_delete = [self._casebase[key] for key in data if key in self._casebase]

        for key in data:
            self._casebase.pop(key, None)

        if texts_to_delete:
            self.conversion_func.delete_index(texts_to_delete)

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
    import Stemmer

    @dataclass(slots=True)
    class bm25[K](
        RetrieverFunc[K, str, float], IndexableFunc[Casebase[K, str], Collection[K]]
    ):
        """BM25 retriever based on bm25s.

        Args:
            language: Language for stemming.
            stopwords: Stopword configuration. ``None`` (default) uses the
                ``language`` for built-in stopwords, a ``str`` overrides the
                stopwords language independently, and a ``list[str]`` provides
                custom stopwords.
            normalize_scores: If ``True`` (default), apply per-query min-max
                normalization to BM25 scores. If ``False``, return raw BM25
                scores.

        Pass an empty casebase to ``__call__`` to use the pre-indexed casebase.
        """

        language: str
        stopwords: str | list[str] | None = None
        normalize_scores: bool = True
        _casebase: dict[K, str] | None = field(default=None, init=False, repr=False)
        _retriever: bm25s.BM25 | None = field(default=None, init=False, repr=False)

        @property
        def _stopwords(self) -> str | list[str]:
            return self.stopwords if self.stopwords is not None else self.language

        @property
        def _stemmer(self) -> Stemmer.Stemmer:
            return Stemmer.Stemmer(self.language)

        def _build_retriever(self, casebase: Casebase[K, str]) -> bm25s.BM25:
            cases_tokens = bm25s.tokenize(
                list(casebase.values()),
                stemmer=self._stemmer,
                stopwords=self._stopwords,
            )
            retriever = bm25s.BM25()
            retriever.index(cases_tokens)

            return retriever

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase."""
            if self._casebase is None:
                return {}
            return self._casebase

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild BM25 index, skipping rebuild when data is unchanged."""
            if self._casebase is not None and dict(data) == self._casebase:
                return

            self._casebase = dict(data)
            self._retriever = self._build_retriever(data)

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Merge new data with existing casebase and rebuild index."""
            if self._casebase is None:
                self.create_index(data)
                return

            self._casebase.update(data)
            self._retriever = self._build_retriever(self._casebase)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Remove keys from the casebase and rebuild index."""
            if self._casebase is None:
                return

            for key in data:
                self._casebase.pop(key, None)

            if self._casebase:
                self._retriever = self._build_retriever(self._casebase)
            else:
                self._retriever = None

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
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
        ) -> Sequence[dict[K, float]]:
            if self._retriever and self._casebase is casebase:
                retriever = self._retriever
            else:
                retriever = self._build_retriever(casebase)

            queries_tokens = bm25s.tokenize(
                cast(list[str], queries),
                stemmer=self._stemmer,
                stopwords=self._stopwords,
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
    import lancedb as ldb
    import numpy as np

    @dataclass(slots=True)
    class lancedb[K: int | str](_DatabaseRetriever[K]):
        """Vector database-backed retriever using LanceDB.

        Delegates storage and search to an embedded LanceDB database,
        keeping data on disk instead of in process memory.
        Supports dense (ANN), sparse (FTS/BM25), and hybrid search.

        Args:
            uri: Path to the LanceDB database directory.
            table: Table name within the database.
            search_type: Search mode — ``"dense"`` (ANN), ``"sparse"``
                (BM25), or ``"hybrid"`` (dense + sparse combined).
            conversion_func: Embedding function. Required for ``"dense"``
                and ``"hybrid"`` search types.
            query_conversion_func: Optional separate embedding function
                for queries.
            limit: Maximum number of results to return per query.
                Caps the database-level search to improve performance on
                large tables. ``None`` (default) returns all rows.
            key_column: Column name for case keys.
            value_column: Column name for case text values.
            vector_column: Column name for dense embedding vectors.
            metadata_func: Optional callable that produces extra columns
                for each row. Called with ``(key, value)`` and must return
                a dict mapping column names to values.
            where: Optional SQL filter expression applied during every
                search query (e.g. ``"category = 'A'"``).
            normalize_scores: Apply per-query min-max normalization.

        Pass an empty casebase to ``__call__`` to use the indexed database
        state. If a table with the given name already exists at ``uri``,
        it is opened automatically on init and can be queried immediately.
        """

        uri: str
        table: str
        search_type: Literal["dense", "sparse", "hybrid"] = "dense"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        limit: int | None = None
        key_column: str = "key"
        value_column: str = "value"
        vector_column: str = "vector"
        metadata_func: Callable[[K, str], dict[str, Any]] | None = None
        where: str | None = None
        normalize_scores: bool = True
        _db: ldb.DBConnection = field(init=False, repr=False)
        _table: ldb.Table | None = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            if self.search_type in ("dense", "hybrid") and self.conversion_func is None:
                raise ValueError(
                    f"conversion_func is required for search_type={self.search_type!r}"
                )

            self._db = ldb.connect(self.uri)

            if self.table in self._db.list_tables().tables:
                self._table = self._db.open_table(self.table)

        def _search_limit(self) -> int | None:
            """Return the effective search limit, or ``None`` for unlimited."""
            if self._table is None:
                return self.limit

            n = self._table.count_rows()

            if self.limit is not None:
                return min(self.limit, n)

            return n

        def _build_rows(self, casebase: Casebase[K, str]) -> list[dict[str, Any]]:
            """Build row dicts for LanceDB from a casebase."""
            keys = list(casebase.keys())
            values = list(casebase.values())

            if self.search_type == "sparse":
                rows = [
                    {self.key_column: key, self.value_column: value}
                    for key, value in zip(keys, values, strict=True)
                ]
            else:
                assert self.conversion_func is not None
                vecs = self.conversion_func(values)
                rows = [
                    {
                        self.key_column: key,
                        self.value_column: value,
                        self.vector_column: np.asarray(vec).tolist(),
                    }
                    for key, value, vec in zip(keys, values, vecs, strict=True)
                ]

            if self.metadata_func is not None:
                for row, key, value in zip(rows, keys, values, strict=True):
                    row.update(self.metadata_func(key, value))

            return rows

        def _setup_indices(self, table: ldb.Table) -> None:
            """Create scalar and optional FTS indices on a table."""
            table.create_scalar_index(self.key_column, replace=True)

            if self.search_type in ("sparse", "hybrid"):
                table.create_fts_index(self.value_column, replace=True)

        @override
        def _has_index(self) -> bool:
            return self._table is not None

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase from the LanceDB table."""
            if self._table is None:
                return {}
            table = self._table.to_arrow()
            keys = table.column(self.key_column).to_pylist()
            values = table.column(self.value_column).to_pylist()
            return dict(zip(keys, values, strict=True))

        @override
        def _search_brute(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
        ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
            if self.search_type != "dense":
                raise NotImplementedError(
                    f"Brute-force search is not supported for search_type={self.search_type!r}. "
                    "Call create_index() first."
                )

            assert self.conversion_func is not None

            keys = list(casebase.keys())
            values = list(casebase.values())

            case_vecs = self.conversion_func(values)
            embed_func = self.query_conversion_func or self.conversion_func
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

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild LanceDB table, reusing existing rows where possible."""
            if not data or self._table is None:
                rows = self._build_rows(data)
                table = self._db.create_table(self.table, rows, mode="overwrite")
                self._setup_indices(table)
                self._table = table
                return

            existing = self.index
            new_keys = set(data.keys())
            old_keys = set(existing.keys())

            stale_keys = old_keys - new_keys
            changed_or_new: Casebase[K, str] = {
                k: data[k]
                for k in new_keys
                if k not in existing or existing[k] != data[k]
            }

            if not stale_keys and not changed_or_new:
                return

            keys_to_delete = stale_keys | (set(changed_or_new.keys()) & old_keys)
            if keys_to_delete:
                self.delete_index(keys_to_delete)

            if changed_or_new:
                rows = self._build_rows(changed_or_new)
                self._table.add(rows)

            self._setup_indices(self._table)

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Append rows to an existing LanceDB table."""
            if self._table is None:
                self.create_index(data)
                return

            rows = self._build_rows(data)
            self._table.add(rows)
            self._setup_indices(self._table)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Delete rows from the LanceDB table by key."""
            if self._table is None:
                return

            if not data:
                return

            key_list = list(data)
            sample = key_list[0]
            col = self.key_column

            if isinstance(sample, str):
                predicate = f"{col} IN (" + ", ".join(f"'{k}'" for k in key_list) + ")"
            else:
                predicate = f"{col} IN (" + ", ".join(str(k) for k in key_list) + ")"

            self._table.delete(predicate)
            self._setup_indices(self._table)

        def _hits_to_result(
            self,
            hits: list[dict[str, Any]],
            score_key: str,
        ) -> tuple[Casebase[K, str], SimMap[K, float]]:
            """Convert LanceDB hit dicts to a (casebase, score_map) pair."""
            return (
                {hit[self.key_column]: hit[self.value_column] for hit in hits},
                {hit[self.key_column]: float(hit[score_key]) for hit in hits},
            )

        @override
        def _search_db_dense(
            self,
            queries: Sequence[str],
        ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
            assert self._table is not None
            assert self.conversion_func is not None

            embed_func = self.query_conversion_func or self.conversion_func
            query_vecs = embed_func(list(queries))
            n = self._search_limit()

            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for qvec in query_vecs:
                q = self._table.search(
                    np.asarray(qvec).tolist(),
                    vector_column_name=self.vector_column,
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
            assert self._table is not None

            n = self._search_limit()
            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for query in queries:
                q = self._table.search(query, query_type="fts").limit(n)

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
            assert self._table is not None
            assert self.conversion_func is not None

            embed_func = self.query_conversion_func or self.conversion_func
            query_vecs = embed_func(list(queries))
            n = self._search_limit()

            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for query, qvec in zip(queries, query_vecs, strict=True):
                q = (
                    self._table.search(query_type="hybrid")
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
    from chromadb.api import ClientAPI
    from chromadb.api.types import SearchResult

    @dataclass(slots=True)
    class chromadb[K: str](_DatabaseRetriever[K]):
        """ChromaDB-backed retriever with dense, sparse, and hybrid search.

        Embedding is handled entirely by ChromaDB's embedding function
        protocols.  Pass a ChromaDB ``EmbeddingFunction`` for dense search
        and/or a ``SparseEmbeddingFunction`` for sparse search.

        Uses ChromaDB's ``Search`` API with ``Knn`` for dense/sparse
        ranking and ``Rrf`` (Reciprocal Rank Fusion) for hybrid search.

        Args:
            path: Directory for PersistentClient storage.
            collection: Collection name.
            search_type: ``"dense"`` (ANN), ``"sparse"`` (BM25/SPLADE),
                or ``"hybrid"`` (dense + sparse combined via RRF).
            embedding_func: ChromaDB ``EmbeddingFunction`` for dense
                embeddings. Required for ``"dense"`` and ``"hybrid"``.
            sparse_embedding_func: ChromaDB ``SparseEmbeddingFunction``
                for sparse embeddings. Required for ``"sparse"`` and
                ``"hybrid"``.
            limit: Max results per query (``None`` = all).
            metadata_func: Produces extra metadata per document from
                ``(key, value)``.
            sparse_key: Key name for the sparse vector index in the
                ChromaDB schema.
            rrf_k: Smoothing parameter for RRF (default: 60).
            rrf_weights: Weights for ``(dense, sparse)`` in RRF
                (default: ``(0.7, 0.3)``).
            normalize_scores: Apply per-query min-max normalization.
        """

        path: str
        collection: str
        search_type: Literal["dense", "sparse", "hybrid"] = "dense"
        embedding_func: cdb.EmbeddingFunction | None = None
        sparse_embedding_func: cdb.SparseEmbeddingFunction | None = None
        limit: int | None = None
        metadata_func: Callable[[K, str], cdb.Metadata] | None = None
        sparse_key: str = "sparse_embedding"
        rrf_k: int = 60
        rrf_weights: tuple[float, float] = field(default=(0.7, 0.3))
        normalize_scores: bool = True
        _client: ClientAPI = field(init=False, repr=False)
        _collection: cdb.Collection | None = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            if self.search_type in ("dense", "hybrid") and self.embedding_func is None:
                raise ValueError(
                    f"embedding_func is required for search_type={self.search_type!r}"
                )
            if (
                self.search_type in ("sparse", "hybrid")
                and self.sparse_embedding_func is None
            ):
                raise ValueError(
                    f"sparse_embedding_func is required for search_type={self.search_type!r}"
                )

            self._client = cdb.PersistentClient(path=self.path)

            try:
                self._collection = self._client.get_collection(
                    self.collection,
                    embedding_function=self.embedding_func,
                )
            except Exception:
                self._collection = None

        def _build_schema(self) -> cdb.Schema | None:
            """Build collection schema with sparse vector index if needed."""
            if self.search_type not in ("sparse", "hybrid"):
                return None

            return cdb.Schema().create_index(
                key=self.sparse_key,
                config=cdb.SparseVectorIndexConfig(
                    embedding_function=self.sparse_embedding_func,
                    source_key=cdb.K.DOCUMENT.name,
                ),
            )

        @override
        def _has_index(self) -> bool:
            return self._collection is not None

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase from the ChromaDB collection."""
            if self._collection is None:
                return {}
            result = self._collection.get()
            ids = result["ids"]
            docs = result["documents"] or []
            return {cast(K, id_): doc for id_, doc in zip(ids, docs, strict=True)}

        def _prepare_documents(
            self,
            data: Casebase[K, str],
        ) -> tuple[list[str], list[str], list[cdb.Metadata] | None]:
            """Prepare IDs, documents, and metadatas from a casebase."""
            ids = [str(k) for k in data.keys()]
            values = list(data.values())
            metadatas: list[cdb.Metadata] | None = None

            if self.metadata_func is not None:
                metadatas = [
                    self.metadata_func(k, v)
                    for k, v in zip(data.keys(), values, strict=True)
                ]

            return ids, values, metadatas

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild ChromaDB collection, reusing existing documents where possible."""
            if self._collection is None:
                collection = self._client.create_collection(
                    name=self.collection,
                    schema=self._build_schema(),
                    embedding_function=self.embedding_func,
                )

                if data:
                    ids, documents, metadatas = self._prepare_documents(data)
                    collection.add(ids=ids, documents=documents, metadatas=metadatas)

                self._collection = collection
                return

            existing = self.index
            new_keys = set(data.keys())
            old_keys = set(existing.keys())

            stale_keys = old_keys - new_keys
            changed_or_new: Casebase[K, str] = {
                k: data[k]
                for k in new_keys
                if k not in existing or existing[k] != data[k]
            }

            if not stale_keys and not changed_or_new:
                return

            if stale_keys:
                self.delete_index(stale_keys)

            if changed_or_new:
                ids, documents, metadatas = self._prepare_documents(changed_or_new)
                self._collection.upsert(
                    ids=ids, documents=documents, metadatas=metadatas
                )

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Upsert documents into an existing ChromaDB collection."""
            if self._collection is None:
                self.create_index(data)
                return

            ids, documents, metadatas = self._prepare_documents(data)
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Remove documents by ID from the ChromaDB collection."""
            if self._collection is None:
                return

            ids = [str(k) for k in data]

            if ids:
                self._collection.delete(ids=ids)

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
                    return cdb.Knn(query=query, key=self.sparse_key, limit=n)
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
                                key=self.sparse_key,
                                return_rank=True,
                                limit=n,
                                default=default_rank,
                            ),
                        ],
                        weights=list(self.rrf_weights),
                        k=self.rrf_k,
                    )

        @override
        def _search_db(
            self,
            queries: Sequence[str],
        ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
            assert self._collection is not None

            n = self.limit or self._collection.count()

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

            return self._normalize_results(
                self._process_result(
                    self._collection.search(searches),
                    score_transform=score_transform,
                )
            )
