from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from inspect import signature as inspect_signature
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
    RetrieverPairFunc,
    RetrieverSeqFunc,
    SimMap,
    SimSeq,
    SupportsMetadata,
    SupportsParallelQueries,
)

__all__ = [
    "build",
    "apply_queries",
    "apply_query",
    "Result",
    "ResultStep",
    "base_retriever",
]


@dataclass(slots=True, frozen=True)
class QueryResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]

    @classmethod
    def build(
        cls, similarities: Mapping[K, tuple[S, bool]], full_casebase: Casebase[K, V]
    ) -> "QueryResultStep[K, V, S]":
        filtered_sims = {}
        all_sims = {}

        for key, value in similarities.items():
            sim, passed_filter = value
            all_sims[key] = sim

            if passed_filter:
                filtered_sims[key] = sim

        ranking = similarities2ranking(filtered_sims)
        casebase = {key: full_casebase[key] for key in filtered_sims}

        return cls(all_sims, tuple(ranking), casebase)

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
    def default_query(self) -> QueryResultStep[C, V, S]:
        return self.final_step.default_query

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.default_query.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.final_step.default_query.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.final_step.default_query.casebase

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
        >>> query = casebase[42]
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
        >>> result = cbrkit.retrieval.apply(casebase, query, retriever)
    """
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    steps: list[ResultStep[Q, C, V, S]] = []
    current_casebases: Mapping[Q, Mapping[C, V]] = {
        query_key: casebase for query_key in queries
    }

    for retriever_func in retrievers:
        retriever_signature = inspect_signature(retriever_func)

        if len(retriever_signature.parameters) == 2:
            retriever_func = cast(RetrieverPairFunc[C, V, S], retriever_func)

            if (
                isinstance(retriever_func, SupportsParallelQueries)
                and retriever_func.query_processes > 1
            ):
                pool_processes = (
                    None
                    if retriever_func.query_processes <= 0
                    else retriever_func.query_processes
                )

                with Pool(pool_processes) as pool:
                    sim_maps = pool.starmap(
                        retriever_func,
                        (
                            (current_casebases[query_key], query)
                            for query_key, query in queries.items()
                        ),
                    )
                    step_queries = {
                        query_key: QueryResultStep.build(sim_map, casebase)
                        for query_key, sim_map in zip(queries, sim_maps, strict=True)
                    }
            else:
                step_queries = {
                    query_key: QueryResultStep.build(
                        retriever_func(current_casebases[query_key], query), casebase
                    )
                    for query_key, query in queries.items()
                }

        else:
            retriever_func = cast(RetrieverSeqFunc[V, S], retriever_func)
            similarities = retriever_func(
                [
                    (case, query)
                    for query_key, query in queries.items()
                    for case in current_casebases[query_key].values()
                ]
            )

            step_queries = {
                query_key: QueryResultStep.build(
                    {
                        case_key: similarities[
                            case_idx + query_idx * len(current_casebases[query_key])
                        ]
                        for case_idx, case_key in enumerate(
                            current_casebases[query_key]
                        )
                    },
                    current_casebases[query_key],
                )
                for query_idx, query_key in enumerate(queries)
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


def chunkify_number[V](val: Sequence[V], n: int) -> Iterator[Sequence[V]]:
    """Yield successive n-sized chunks from val.

    Args:
        val: The sequence that will be chunked.
        n: Number of complete chunks.

    Returns:
        An iterator that yields the chunks.

    Examples:
        >>> list(chunkify_number([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2], [3, 4], [5, 6], [7, 8], [9]]
    """

    k = len(val) // n
    return chunkify_size(val, k)


def chunkify_size[V](val: Sequence[V], k: int) -> Iterator[Sequence[V]]:
    """Yield a total of k chunks from val.

    Args:
        val: The sequence that will be chunked.
        k: Number of chunks.

    Returns:
        An iterator that yields the chunks.

    Examples:
        >>> list(chunkify_size([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """

    for i in range(0, len(val), k):
        yield val[i : i + k]


@dataclass(slots=True, frozen=True, kw_only=True)
class base_retriever[K, S: Float](SupportsMetadata):
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

    def filter_ranking(self, similarities: SimSeq[S] | SimMap[K, S]) -> list[Any]:
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
        if self.limit is None:
            ranking = ranking[: self.limit]

        return ranking

    def process_seq(self, similarities: SimSeq[S]) -> Sequence[tuple[S, bool]]:
        ranking = self.filter_ranking(similarities)

        return [
            (sim, True) if key in ranking else (sim, False)
            for key, sim in enumerate(similarities)
        ]

    def process_map(self, similarities: SimMap[K, S]) -> Mapping[K, tuple[S, bool]]:
        ranking = self.filter_ranking(similarities)

        return {
            key: (sim, True) if key in ranking else (sim, False)
            for key, sim in similarities.items()
        }


@dataclass(slots=True, frozen=True)
class build[V, S: Float](base_retriever[V, S], RetrieverSeqFunc[V, S]):
    """Based on the similarity function this function creates a retriever function.

    The given limit will be applied after filtering for min/max similarity.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
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
    chunk_size: int = 100

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            **super(build, self).metadata,
            "similarity_func": get_metadata(self.similarity_func),
            "processes": self.processes,
            "chunk_size": self.chunk_size,
        }

    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> Sequence[tuple[S, bool]]:
        sim_func = SimSeqWrapper(self.similarity_func)
        similarities: Sequence[S] = []

        if self.processes != 1:
            pool_processes = None if self.processes <= 0 else self.processes
            pair_chunks = chunkify_size(pairs, self.chunk_size)

            with Pool(pool_processes) as pool:
                sim_chunks = pool.starmap(sim_func, pair_chunks)

            for sim_chunk in sim_chunks:
                similarities.extend(sim_chunk)
        else:
            similarities = sim_func(pairs)

        return self.process_seq(similarities)


try:
    from cohere import Client
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere[K, V](
        base_retriever[K, float],
        RetrieverPairFunc[K, V, float],
        SupportsParallelQueries,
    ):
        """Semantic similarity using Cohere's rerank models

        Args:
            model: Name of the [rerank model](https://docs.cohere.com/reference/rerank).
        """

        model: str
        conversion_func: Callable[[V], str]
        top_n: int | None = None
        max_chunks_per_doc: int | None = None
        client: Client = field(default_factory=Client)
        request_options: RequestOptions | None = None

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                **super(cohere, self).metadata,
                "model": self.model,
                "conversion_func": get_metadata(self.conversion_func),
                "top_n": self.top_n,
                "max_chunks_per_doc": self.max_chunks_per_doc,
                "request_options": str(self.request_options),
            }

        @override
        def __call__(
            self,
            casebase: Casebase[K, V],
            query: V,
        ) -> Casebase[K, tuple[float, bool]]:
            response = self.client.v2.rerank(
                model=self.model,
                query=self.conversion_func(query),
                documents=[self.conversion_func(value) for value in casebase.values()],
                return_documents=False,
                top_n=self.top_n,
                max_chunks_per_doc=self.max_chunks_per_doc,
                request_options=self.request_options,
            )
            key_index = {idx: key for idx, key in enumerate(casebase)}

            similarities: SimMap[K, float] = {
                key_index[result.index]: result.relevance_score
                for result in response.results
            }

            return self.process_map(similarities)

    __all__ += ["cohere"]

except ImportError:
    pass
