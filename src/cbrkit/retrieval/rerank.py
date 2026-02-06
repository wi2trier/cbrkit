import asyncio
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from frozendict import frozendict

from ..helpers import (
    batchify_sim,
    dispatch_batches,
    get_logger,
    optional_dependencies,
    run_coroutine,
)
from ..sim.embed import cache, default_score_func
from ..typing import (
    AnySimFunc,
    BatchConversionFunc,
    BatchSimFunc,
    Casebase,
    Float,
    HasMetadata,
    JsonDict,
    NumpyArray,
    RetrieverFunc,
)
from .model import IndexableRetrieverFunc

logger = get_logger(__name__)

with optional_dependencies():
    from cohere import AsyncClient
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere[K](RetrieverFunc[K, str, float]):
        """Semantic similarity using Cohere's rerank models

        Args:
            model: Name of the [rerank model](https://docs.cohere.com/reference/rerank).
        """

        model: str
        max_tokens_per_doc: int | None = None
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)
        request_options: RequestOptions | None = field(default=None, repr=False)

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return run_coroutine(self._retrieve(batches))

        async def _retrieve(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in batches)
            )

        async def _retrieve_single(
            self,
            query: str,
            casebase: Casebase[K, str],
        ) -> dict[K, float]:
            response = await self.client.v2.rerank(
                model=self.model,
                query=query,
                documents=list(casebase.values()),
                max_tokens_per_doc=self.max_tokens_per_doc,
                request_options=self.request_options,
            )
            key_index = {idx: key for idx, key in enumerate(casebase)}

            return {
                key_index[result.index]: result.relevance_score
                for result in response.results
            }


with optional_dependencies():
    from voyageai.client_async import AsyncClient

    @dataclass(slots=True, frozen=True)
    class voyageai[K](RetrieverFunc[K, str, float]):
        """Semantic similarity using Voyage AI's rerank models

        Args:
            model: Name of the [rerank model](https://docs.voyageai.com/docs/reranker).
        """

        model: str
        truncation: bool = True
        client: AsyncClient = field(default_factory=AsyncClient, repr=False)

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return run_coroutine(self._retrieve(batches))

        async def _retrieve(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            return await asyncio.gather(
                *(self._retrieve_single(query, casebase) for casebase, query in batches)
            )

        async def _retrieve_single(
            self,
            query: str,
            casebase: Casebase[K, str],
        ) -> dict[K, float]:
            response = await self.client.rerank(
                model=self.model,
                query=query,
                documents=list(casebase.values()),
                truncation=self.truncation,
            )
            key_index = {idx: key for idx, key in enumerate(casebase)}

            return {
                key_index[result.index]: result.relevance_score
                for result in response.results
            }


with optional_dependencies():
    from sentence_transformers import SentenceTransformer, util

    @dataclass(slots=True, frozen=True)
    class sentence_transformers[K](
        RetrieverFunc[K, str, float],
        HasMetadata,
    ):
        """Semantic similarity using sentence transformers

        Args:
            model: Name of the [sentence transformer model](https://www.sbert.net/docs/pretrained_models.html).
        """

        model: SentenceTransformer | str
        query_chunk_size: int = 100
        corpus_chunk_size: int = 500000
        device: str | None = None

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "model": self.model if isinstance(self.model, str) else "custom",
                "query_chunk_size": self.query_chunk_size,
                "corpus_chunk_size": self.corpus_chunk_size,
                "device": self.device,
            }

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[dict[K, float]]:
            if not batches:
                return []

            if isinstance(self.model, str):
                model = SentenceTransformer(self.model, device=self.device)  # pyright: ignore
            else:
                model = self.model

            model.to(self.device)

            return dispatch_batches(
                batches,
                lambda queries, casebase: self.__call_queries__(queries, casebase, model),
            )

        def __call_queries__(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
            model: SentenceTransformer,
        ) -> Sequence[dict[K, float]]:
            case_texts = list(casebase.values())
            query_texts = cast(list[str], queries)

            case_embeddings = util.normalize_embeddings(
                model.encode(case_texts, convert_to_tensor=True).to(self.device)
            )
            query_embeddings = util.normalize_embeddings(
                model.encode(query_texts, convert_to_tensor=True).to(self.device)
            )

            response = util.semantic_search(
                query_embeddings,
                case_embeddings,
                top_k=len(casebase),
                query_chunk_size=self.query_chunk_size,
                corpus_chunk_size=self.corpus_chunk_size,
                score_function=util.dot_score,
            )

            key_index = {idx: key for idx, key in enumerate(casebase)}

            return [
                {
                    key_index[cast(int, res["corpus_id"])]: cast(float, res["score"])
                    for res in query_response
                }
                for query_response in response
            ]


with optional_dependencies():
    import bm25s
    import numpy as np
    import Stemmer

    @dataclass(slots=True)
    class bm25[K](IndexableRetrieverFunc[K, str, float]):
        """BM25 retriever based on bm25s"""

        language: str
        stopwords: list[str] | None = None
        casebase: Casebase[K, str] | None = field(default=None, init=False, repr=False)
        _retriever: bm25s.BM25 | None = field(default=None, init=False, repr=False)

        @property
        def _stopwords(self) -> str | list[str]:
            return self.stopwords or self.language

        @property
        def _stemmer(self) -> Callable[..., Any]:
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

        @override
        def index(self, casebase: Casebase[K, str], prune: bool = True) -> None:
            if not prune:
                raise NotImplementedError("BM25 requires pruning")

            if not isinstance(casebase, frozendict):
                raise TypeError("BM25 casebase must be a frozendict")

            self._retriever = self._build_retriever(casebase)
            self.casebase = casebase

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[dict[K, float]]:
            return dispatch_batches(batches, self.__call_queries__)

        def __call_queries__(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
        ) -> Sequence[dict[K, float]]:
            # identity check can only be used because `index` enforces frozendict
            if self._retriever and self.casebase is casebase:
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
            max_score = np.max(scores)
            min_score = np.min(scores)
            score_range = max_score - min_score

            key_index = {idx: key for idx, key in enumerate(casebase)}

            return [
                {
                    key_index[case_id]: float(
                        (score - min_score) / score_range if score_range != 0 else 1.0
                    )
                    for case_id, score in zip(
                        results[query_id], scores[query_id], strict=True
                    )
                }
                for query_id in range(len(queries))
            ]


@dataclass(slots=True, init=False)
class embed[K, S: Float](IndexableRetrieverFunc[K, str, S]):
    """Embedding-based semantic retriever with indexing support.

    Args:
        conversion_func: Embedding function (from embed module).
        sim_func: Vector similarity function (default: cosine).
        query_conversion_func: Optional separate embedding function for queries.
    """

    conversion_func: cache
    sim_func: BatchSimFunc[NumpyArray, S]
    query_conversion_func: cache | None
    casebase: Casebase[K, str] | None = field(repr=False, init=False, default=None)

    def __init__(
        self,
        conversion_func: cache,
        sim_func: AnySimFunc[NumpyArray, S] = default_score_func,
        query_conversion_func: cache | None = None,
    ):
        self.conversion_func = conversion_func
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = query_conversion_func
        self.casebase = None

    @override
    def index(self, casebase: Casebase[K, str], prune: bool = True) -> None:
        if not prune and self.casebase:
            self.casebase = {**self.casebase, **casebase}
        else:
            self.casebase = casebase

        self.conversion_func.index(casebase.values(), prune=prune)

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[dict[K, S]]:
        if not batches:
            return []

        # Collect all texts for embedding
        resolved_batches: list[tuple[Casebase[K, str], str]] = []
        all_case_texts: list[str] = []
        all_query_texts: list[str] = []

        for casebase, query in batches:
            resolved_batches.append((casebase, query))
            all_case_texts.extend(casebase.values())
            all_query_texts.append(query)

        # Compute all embeddings in one batch
        if self.query_conversion_func:
            all_case_vecs = self.conversion_func(all_case_texts)
            all_query_vecs = self.query_conversion_func(all_query_texts)
        else:
            # Batch all texts together when using the same conversion function
            all_texts = all_case_texts + all_query_texts
            all_vecs = self.conversion_func(all_texts)
            all_case_vecs = all_vecs[: len(all_case_texts)]
            all_query_vecs = all_vecs[len(all_case_texts) :]

        # Build results using batched similarity
        results: list[dict[K, S]] = []
        case_vec_idx = 0

        for query_idx, (casebase, _) in enumerate(resolved_batches):
            case_keys = list(casebase.keys())
            case_vecs = all_case_vecs[case_vec_idx : case_vec_idx + len(case_keys)]
            case_vec_idx += len(case_keys)
            query_vec = all_query_vecs[query_idx]

            sims = self.sim_func([(cv, query_vec) for cv in case_vecs])
            results.append(dict(zip(case_keys, sims, strict=True)))

        return results


with optional_dependencies():
    import lancedb as ldb
    import numpy as np

    @dataclass(slots=True, frozen=True)
    class LancedbCasebase[K: int | str](Mapping[K, str]):
        """Casebase backed by a LanceDB table.

        Keys and values are stored in the table itself.
        """

        store: ldb.Table

        @override
        def __getitem__(self, key: K) -> str:
            ds = self.store.to_lance()
            result = ds.to_table(filter=f"key = {key!r}", columns=["value"])

            if len(result) == 0:
                raise KeyError(key)

            return cast(str, result.column("value")[0].as_py())

        @override
        def __iter__(self) -> Iterator[K]:
            ds = self.store.to_lance()
            keys: list[K] = ds.to_table(columns=["key"]).column("key").to_pylist()
            return iter(keys)

        @override
        def __len__(self) -> int:
            return self.store.count_rows()

    @dataclass(slots=True)
    class lancedb[K: int | str](IndexableRetrieverFunc[K, str, float]):
        """Vector database-backed retriever using LanceDB.

        Delegates storage and search to an embedded LanceDB database,
        keeping data on disk instead of in process memory.
        Supports vector (ANN), full-text (FTS/BM25), and hybrid search.

        Args:
            uri: Path to the LanceDB database directory.
            table: Table name within the database.
            search_type: Search mode — ``"vector"`` (ANN), ``"fts"`` (BM25),
                or ``"hybrid"`` (vector + FTS combined).
            conversion_func: Embedding function. Required for ``"vector"``
                and ``"hybrid"`` search types.
            query_conversion_func: Optional separate embedding function for queries.
        """

        uri: str
        table: str
        search_type: Literal["vector", "fts", "hybrid"] = "vector"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        query_conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        casebase: Casebase[K, str] | None = field(
            default=None, init=False, repr=False
        )

        def __post_init__(self) -> None:
            if self.search_type in ("vector", "hybrid") and self.conversion_func is None:
                raise ValueError(
                    f"conversion_func is required for search_type={self.search_type!r}"
                )

        @override
        def index(self, casebase: Casebase[K, str], prune: bool = True) -> None:
            keys = list(casebase.keys())
            values = list(casebase.values())

            if self.search_type == "fts":
                rows = [
                    {"key": key, "value": value}
                    for key, value in zip(keys, values, strict=True)
                ]
            else:
                assert self.conversion_func is not None
                vecs = self.conversion_func(values)
                rows = [
                    {
                        "key": key,
                        "value": value,
                        "vector": np.asarray(vec).tolist(),
                    }
                    for key, value, vec in zip(keys, values, vecs, strict=True)
                ]

            db = ldb.connect(self.uri)

            if prune or self.table not in db.table_names():
                table = db.create_table(self.table, rows, mode="overwrite")
            else:
                table = db.open_table(self.table)
                table.add(rows)

            table.create_scalar_index("key", replace=True)

            if self.search_type in ("fts", "hybrid"):
                table.create_fts_index("value", replace=True)

            self.casebase = LancedbCasebase(store=table)

        @override
        def __call__(
            self,
            batches: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[dict[K, float]]:
            return dispatch_batches(batches, self._call_queries)

        def _call_queries(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
        ) -> Sequence[dict[K, float]]:
            if self.casebase is not None and self.casebase is casebase:
                if self.search_type == "vector":
                    return self._search_db_vector(queries)
                elif self.search_type == "fts":
                    return self._search_db_fts(queries)
                else:
                    return self._search_db_hybrid(queries)

            return self._search_brute(queries, casebase)

        def _search_db_vector(
            self,
            queries: Sequence[str],
        ) -> Sequence[dict[K, float]]:
            assert isinstance(self.casebase, LancedbCasebase)
            assert self.conversion_func is not None

            embed_func = self.query_conversion_func or self.conversion_func
            query_vecs = embed_func(list(queries))
            n = len(self.casebase)

            results: list[dict[K, float]] = []

            for qvec in query_vecs:
                hits = (
                    self.casebase.store.search(np.asarray(qvec).tolist())
                    .limit(n)
                    .to_list()
                )
                results.append(
                    {hit["key"]: 1.0 - hit["_distance"] for hit in hits}
                )

            return results

        def _search_db_fts(
            self,
            queries: Sequence[str],
        ) -> Sequence[dict[K, float]]:
            assert isinstance(self.casebase, LancedbCasebase)

            n = len(self.casebase)
            results: list[dict[K, float]] = []

            for query in queries:
                hits = (
                    self.casebase.store.search(query, query_type="fts")
                    .limit(n)
                    .to_list()
                )

                if not hits:
                    results.append({})
                    continue

                scores = [hit["_score"] for hit in hits]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                results.append(
                    {
                        hit["key"]: (
                            (hit["_score"] - min_score) / score_range
                            if score_range != 0
                            else 1.0
                        )
                        for hit in hits
                    }
                )

            return results

        def _search_db_hybrid(
            self,
            queries: Sequence[str],
        ) -> Sequence[dict[K, float]]:
            assert isinstance(self.casebase, LancedbCasebase)
            assert self.conversion_func is not None

            embed_func = self.query_conversion_func or self.conversion_func
            query_vecs = embed_func(list(queries))
            n = len(self.casebase)

            results: list[dict[K, float]] = []

            for query, qvec in zip(queries, query_vecs, strict=True):
                hits = (
                    self.casebase.store.search(query_type="hybrid")
                    .vector(np.asarray(qvec).tolist())
                    .text(query)
                    .limit(n)
                    .to_list()
                )

                if not hits:
                    results.append({})
                    continue

                scores = [hit["_relevance_score"] for hit in hits]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score

                results.append(
                    {
                        hit["key"]: (
                            (hit["_relevance_score"] - min_score) / score_range
                            if score_range != 0
                            else 1.0
                        )
                        for hit in hits
                    }
                )

            return results

        def _search_brute(
            self,
            queries: Sequence[str],
            casebase: Casebase[K, str],
        ) -> Sequence[dict[K, float]]:
            if self.search_type in ("fts", "hybrid"):
                raise NotImplementedError(
                    f"Brute-force search is not supported for search_type={self.search_type!r}. "
                    "Call index() first."
                )

            assert self.conversion_func is not None

            keys = list(casebase.keys())
            values = list(casebase.values())

            case_vecs = self.conversion_func(values)
            embed_func = self.query_conversion_func or self.conversion_func
            query_vecs = embed_func(list(queries))

            sim_func = batchify_sim(default_score_func)

            results: list[dict[K, float]] = []

            for qvec in query_vecs:
                sims = sim_func([(cv, qvec) for cv in case_vecs])
                results.append(dict(zip(keys, sims, strict=True)))

            return results
