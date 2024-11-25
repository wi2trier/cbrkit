from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from typing import Any, cast, override

from .helpers import (
    SimSeqWrapper,
    get_metadata,
    similarities2ranking,
    unpack_sim,
)
from .typing import (
    AnySimFunc,
    Casebase,
    Float,
    JsonDict,
    RetrieverFunc,
    SimMap,
    SimSeq,
    SimSeqOrMap,
    SupportsMetadata,
)

__all__ = [
    "build",
    "apply_queries",
    "apply_query",
    "Result",
    "ResultStep",
    "base_retriever",
    "transpose",
]


@dataclass(slots=True, frozen=True)
class QueryResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]

    @classmethod
    def build(
        cls, similarities: Mapping[K, S], full_casebase: Casebase[K, V]
    ) -> "QueryResultStep[K, V, S]":
        ranking = similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in similarities}

        return cls(similarities, tuple(ranking), casebase)

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]

        return x


@dataclass(slots=True, frozen=True)
class ResultStep[Q, C, V, S: Float]:
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonDict

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        return next(iter(self.queries.values()))

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.default_query.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.default_query.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.default_query.casebase


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    steps: list[ResultStep[Q, C, V, S]]

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        return self.steps[-1]

    @property
    def metadata(self) -> JsonDict:
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        return self.final_step.queries

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.final_step.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.final_step.casebase

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for step in x["steps"]:
            for item in step["queries"].values():
                del item["casebase"]

        return x


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies a single query to a Casebase using retriever functions.

    Args:
        casebase: The casebase that will be used to retrieve similar cases.
        queries: The queries that will be used to retrieve similar cases.
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Returns an object of type Result.

    Examples:
        >>> import cbrkit
        >>> import polars as pl
        >>> df = pl.read_csv("./data/cars-1k.csv")
        >>> casebase = cbrkit.loaders.polars(df)
        >>> retriever = cbrkit.retrieval.build(
        ...     cbrkit.sim.attribute_value(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.linear(max=100000),
        ...             "year": cbrkit.sim.numbers.linear(max=50),
        ...             "manufacturer": cbrkit.sim.strings.taxonomy.load(
        ...                 "./data/cars-taxonomy.yaml",
        ...                 measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
        ...             ),
        ...             "miles": cbrkit.sim.numbers.linear(max=1000000),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     ),
        ...     limit=5,
        ... )
        >>> result = cbrkit.retrieval.apply_queries(casebase, {"default": casebase[42]}, retriever)
    """
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    steps: list[ResultStep[Q, C, V, S]] = []
    current_casebases: Mapping[Q, Mapping[C, V]] = {
        query_key: casebase for query_key in queries
    }

    for retriever_func in retrievers:
        queries_results = retriever_func(
            [
                (current_casebases[query_key], query)
                for query_key, query in queries.items()
            ]
        )

        step_queries = {
            query_key: QueryResultStep.build(
                retrieved_casebase,
                casebase,
            )
            for query_key, retrieved_casebase in zip(
                queries, queries_results, strict=True
            )
        }

        step = ResultStep(step_queries, get_metadata(retriever_func))
        steps.append(step)
        current_casebases = {
            query_key: step.queries[query_key].casebase for query_key in queries
        }

    return Result(steps)


def apply_query[K, V, S: Float](
    casebase: Mapping[K, V],
    query: V,
    retrievers: RetrieverFunc[K, V, S] | Sequence[RetrieverFunc[K, V, S]],
) -> Result[str, K, V, S]:
    return apply_queries(
        casebase,
        {"default": query},
        retrievers,
    )


apply = apply_query


def chunkify[V](val: Sequence[V], k: int) -> Iterator[Sequence[V]]:
    """Yield a total of k chunks from val.

    Examples:
        >>> list(chunkify([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """

    for i in range(0, len(val), k):
        yield val[i : i + k]


@dataclass(slots=True, frozen=True, kw_only=True)
class base_retriever[K, V, S: Float](SupportsMetadata):
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "limit": self.limit,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }

    def preprocess(
        self,
        similarities: SimSeqOrMap[K, S],
    ) -> list[Any]:
        ranking = similarities2ranking(similarities)

        if self.min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) >= self.min_similarity
            ]
        if self.max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) <= self.max_similarity
            ]
        if self.limit is not None:
            ranking = ranking[: self.limit]

        return ranking

    def postprocess_seq(
        self, similarities: SimSeq[S], index: Sequence[tuple[int, K]], total_pairs: int
    ) -> Sequence[Casebase[K, S]]:
        result: list[dict[K, S]] = [{} for _ in range(total_pairs)]
        ranking = self.preprocess(similarities)

        filtered_sims = [
            sim if key in ranking else None for key, sim in enumerate(similarities)
        ]

        for (idx, key), sim in zip(index, filtered_sims, strict=True):
            if sim is not None:
                result[idx][key] = sim

        return result

    def postprocess_map(self, similarities: SimMap[K, S]) -> dict[K, S]:
        ranking = self.preprocess(similarities)

        return {key: sim for key, sim in similarities.items() if key in ranking}


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](base_retriever[K, V, S], RetrieverFunc[K, V, S]):
    """Based on the similarity function this function creates a retriever function.

    The given limit will be applied after filtering for min/max similarity.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        processes: Number of processes to use. If processes is less than 1, the number returned by os.cpu_count() is used.
        similarity_chunksize: Number of pairs to process in each chunk.
        limit: Retriever function will return the top limit cases.
        min_similarity: Return only cases with a similarity greater or equal than this.
        max_similarity: Return only cases with a similarity less or equal than this.

    Returns:
        Returns the retriever function.

    Examples:
        >>> import cbrkit
        >>> retriever = cbrkit.retrieval.build(
        ...     cbrkit.sim.attribute_value(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.linear(max=100000),
        ...             "year": cbrkit.sim.numbers.linear(max=50),
        ...             "model": cbrkit.sim.attribute_value(
        ...                 attributes={
        ...                     "make": cbrkit.sim.generic.equality(),
        ...                     "manufacturer": cbrkit.sim.strings.taxonomy.load(
        ...                         "./data/cars-taxonomy.yaml",
        ...                         measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
        ...                     ),
        ...                 }
        ...             ),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     ),
        ...     limit=5,
        ... )
    """

    similarity_func: AnySimFunc[V, S]
    processes: int = 1
    similarity_chunksize: int = 1

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            **super(build, self).metadata,
            "similarity_func": get_metadata(self.similarity_func),
            "processes": self.processes,
            "similarity_chunksize": self.similarity_chunksize,
        }

    @override
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[Casebase[K, S]]:
        sim_func = SimSeqWrapper(self.similarity_func)
        similarities: Sequence[S] = []
        pairs_index: list[tuple[int, K]] = []
        flat_pairs: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(pairs):
            for key, case in casebase.items():
                pairs_index.append((idx, key))
                flat_pairs.append((case, query))

        if self.processes != 1:
            pool_processes = None if self.processes <= 0 else self.processes
            pair_chunks = chunkify(flat_pairs, self.similarity_chunksize)

            with Pool(pool_processes) as pool:
                sim_chunks = pool.map(sim_func, pair_chunks)

            for sim_chunk in sim_chunks:
                similarities.extend(sim_chunk)
        else:
            similarities = sim_func(flat_pairs)

        return self.postprocess_seq(similarities, pairs_index, len(pairs))


try:
    from cohere import Client
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere[K](
        base_retriever[K, str, float],
        RetrieverFunc[K, str, float],
    ):
        """Semantic similarity using Cohere's rerank models

        Args:
            model: Name of the [rerank model](https://docs.cohere.com/reference/rerank).
        """

        model: str
        max_chunks_per_doc: int | None = None
        client: Client = field(default_factory=Client)
        request_options: RequestOptions | None = None

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                **super(cohere, self).metadata,
                "model": self.model,
                "max_chunks_per_doc": self.max_chunks_per_doc,
                "request_options": str(self.request_options),
            }

        @override
        def __call__(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            results: list[dict[K, float]] = []

            for casebase, query in pairs:
                response = self.client.v2.rerank(
                    model=self.model,
                    query=query,
                    documents=list(casebase.values()),
                    return_documents=False,
                    max_chunks_per_doc=self.max_chunks_per_doc,
                    request_options=self.request_options,
                )
                key_index = {idx: key for idx, key in enumerate(casebase)}

                similarities = {
                    key_index[result.index]: result.relevance_score
                    for result in response.results
                }
                results.append(self.postprocess_map(similarities))

            return results

    __all__ += ["cohere"]

except ImportError:
    pass


try:
    from voyageai import Client  # type: ignore

    @dataclass(slots=True, frozen=True)
    class voyageai[K](
        base_retriever[K, str, float],
        RetrieverFunc[K, str, float],
    ):
        """Semantic similarity using Voyage AI's rerank models

        Args:
            model: Name of the [rerank model](https://docs.voyageai.com/docs/reranker).
        """

        model: str
        truncation: bool = True
        client: Client = field(default_factory=Client)

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                **super(voyageai, self).metadata,
                "model": self.model,
                "truncation": self.truncation,
            }

        @override
        def __call__(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            results: list[dict[K, float]] = []

            for casebase, query in pairs:
                response = self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=list(casebase.values()),
                    truncation=self.truncation,
                )
                key_index = {idx: key for idx, key in enumerate(casebase)}

                similarities = {
                    key_index[result.index]: result.relevance_score
                    for result in response.results
                }
                results.append(self.postprocess_map(similarities))

            return results

    __all__ += ["voyageai"]

except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer, util

    @dataclass(slots=True, frozen=True)
    class sentence_transformers[K](
        base_retriever[K, str, float],
        RetrieverFunc[K, str, float],
    ):
        """Semantic similarity using sentence transformers

        Args:
            model: Name of the [sentence transformer model](https://www.sbert.net/docs/pretrained_models.html).
        """

        model: SentenceTransformer | str
        query_chunk_size: int = 100
        corpus_chunk_size: int = 500000
        device: str = "cpu"

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                **super(sentence_transformers, self).metadata,
                "model": self.model if isinstance(self.model, str) else "custom",
                "query_chunk_size": self.query_chunk_size,
                "corpus_chunk_size": self.corpus_chunk_size,
                "device": self.device,
            }

        @override
        def __call__(
            self,
            pairs: Sequence[tuple[Casebase[K, str], str]],
        ) -> Sequence[Casebase[K, float]]:
            model = (
                SentenceTransformer(self.model, device=self.device)
                if isinstance(self.model, str)
                else self.model
            )
            results: list[dict[K, float]] = []

            for casebase, query in pairs:
                case_texts = list(casebase.values())
                query_text = query
                embeddings = model.encode(
                    [query_text] + case_texts, convert_to_tensor=True
                )
                embeddings = embeddings.to(self.device)
                embeddings = util.normalize_embeddings(embeddings)
                query_embeddings = embeddings[0:1]
                case_embeddings = embeddings[1:]

                response = util.semantic_search(
                    query_embeddings,
                    case_embeddings,
                    top_k=len(casebase),
                    query_chunk_size=self.query_chunk_size,
                    corpus_chunk_size=self.corpus_chunk_size,
                    score_function=util.dot_score,
                )[0]
                key_index = {idx: key for idx, key in enumerate(casebase)}

                similarities = {
                    key_index[cast(int, res["corpus_id"])]: cast(float, res["score"])
                    for res in response
                }
                results.append(self.postprocess_map(similarities))

            return results

    __all__ += ["sentence_transformers"]

except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class transpose[K, U, V, S: Float](RetrieverFunc[K, V, S], SupportsMetadata):
    """Transforms a retriever function from one type to another.

    Args:
        conversion_func: A function that converts the input values from one type to another.
        retriever_func: The retriever function to be used on the converted values.
    """

    conversion_func: Callable[[V], U]
    retriever_func: RetrieverFunc[K, U, S]

    @override
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[Casebase[K, S]]:
        return self.retriever_func(
            [
                (
                    {
                        key: self.conversion_func(value)
                        for key, value in casebase.items()
                    },
                    self.conversion_func(query),
                )
                for casebase, query in pairs
            ]
        )
