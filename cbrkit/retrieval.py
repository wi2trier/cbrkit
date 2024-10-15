import os
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Any, Literal, cast, override

from cbrkit.helpers import SimMapWrapper, get_metadata, unpack_sim
from cbrkit.loaders import python as load_python
from cbrkit.typing import (
    AnySimFunc,
    Casebase,
    Float,
    JsonDict,
    RetrieverFunc,
    SimMap,
    SupportsMetadata,
)

__all__ = [
    "build",
    "mapply",
    "apply",
    "load",
    "load_map",
    "Result",
    "ResultStep",
]


def _similarities2ranking[K](
    sim_map: SimMap[K, Any],
) -> list[K]:
    return sorted(sim_map, key=lambda key: unpack_sim(sim_map[key]), reverse=True)


@dataclass(slots=True, frozen=True)
class ResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]
    metadata: JsonDict

    @classmethod
    def build(
        cls,
        similarities: SimMap[K, S],
        full_casebase: Casebase[K, V],
        metadata: JsonDict,
    ) -> "ResultStep[K, V, S]":
        ranking = _similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in ranking}

        return cls(similarities, tuple(ranking), casebase, metadata)

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]

        return x


@dataclass(slots=True, frozen=True)
class Result[K, V, S: Float]:
    steps: list[ResultStep[K, V, S]]

    @property
    def final(self) -> ResultStep[K, V, S]:
        return self.steps[-1]

    @property
    def similarities(self) -> SimMap[K, S]:
        return self.final.similarities

    @property
    def ranking(self) -> Sequence[K]:
        return self.final.ranking

    @property
    def casebase(self) -> Casebase[K, V]:
        return self.final.casebase

    @property
    def metadata(self) -> JsonDict:
        return self.final.metadata

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for entry in x["steps"]:
            del entry["casebase"]

        return x


def mapply[K, V, S: Float](
    casebase: Casebase[K, V],
    queries: Mapping[K, V],
    retrievers: RetrieverFunc[K, V, S] | Sequence[RetrieverFunc[K, V, S]],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[K, Result[K, V, S]]:
    """Applies multiple queries to a Casebase using retriever functions.

    Args:
        casebase: The casebase for the query.
        queries: The queries that will be applied to the casebase
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase
        processes: Number of CPUs that will be used for multiprocessing.
            If 1, a regular loop will be used.
            If 0, the number of processes will be equal to the number of CPUs.
            Negative values will be treated as 0.
        parallel: Strategy for parallelization.
            If "queries", each query will be processed in parallel,
            if "casebase" the whole casebase will be processed in parallel.

    Returns:
        Returns an object of type Result.
    """

    if processes != 1 and parallel == "queries":
        pool_processes = None if processes <= 0 else processes
        keys = list(queries.keys())

        with Pool(pool_processes) as pool:
            results = pool.starmap(
                apply,
                ((casebase, queries[key], retrievers) for key in keys),
            )

        return dict(zip(keys, results, strict=True))

    return {
        key: apply(casebase, value, retrievers, processes)
        for key, value in queries.items()
    }


def apply[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    retrievers: RetrieverFunc[K, V, S] | Sequence[RetrieverFunc[K, V, S]],
    processes: int = 1,
) -> Result[K, V, S]:
    """Applies a single query to a Casebase using retriever functions.

    Args:
        casebase: The casebase for the query.
        query: The query that will be applied to the casebase
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase
        processes: Number of CPUs that will be used for multiprocessing.
            If 1, a regular loop will be used.
            If 0, the number of processes will be equal to the number of CPUs.
            Negative values will be treated as 0.

    Returns:
        Returns an object of type Result.

    Examples:
        >>> import cbrkit
        >>> import pandas as pd
        >>> df = pd.read_csv("./data/cars-1k.csv")
        >>> casebase = cbrkit.loaders.dataframe(df)
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
    results: list[ResultStep[K, V, S]] = []
    current_casebase = casebase

    for retriever_func in retrievers:
        sim_map = retriever_func(current_casebase, query, processes)
        result = ResultStep.build(
            sim_map, current_casebase, get_metadata(retriever_func)
        )

        results.append(result)
        current_casebase = result.casebase

    return Result(results)


def _chunkify[V](val: Sequence[V], n: int) -> Iterator[Sequence[V]]:
    """Yield successive n-sized chunks from val.

    Args:
        val: The sequence that will be chunked.
        n: Number of complete chunks.

    Returns:
        An iterator that yields the chunks.

    Examples:
        >>> list(_chunkify([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2], [3, 4], [5, 6], [7, 8], [9]]
    """

    # first compute the chunk size
    k = len(val) // n
    for i in range(0, len(val), k):
        yield val[i : i + k]


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](RetrieverFunc[K, V, S], SupportsMetadata):
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
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @property
    @override
    def metadata(self) -> dict[str, Any]:
        return {
            "similarity_func": get_metadata(self.similarity_func),
            "limit": self.limit,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }

    @override
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        processes: int = 1,
    ) -> SimMap[K, S]:
        sim_func = SimMapWrapper(self.similarity_func)
        similarities: SimMap[K, S] = {}

        if processes != 1:
            pool_processes = None if processes <= 0 else processes
            chunks_num = os.cpu_count() if not pool_processes else pool_processes
            assert chunks_num is not None

            case_chunks: list[Casebase[K, V]] = [
                dict(chunk) for chunk in _chunkify(list(casebase.items()), chunks_num)
            ]

            with Pool(pool_processes) as pool:
                sim_chunks = pool.starmap(
                    sim_func,
                    ((x_chunk, query) for x_chunk in case_chunks),
                )

            for sim_chunk in sim_chunks:
                similarities.update(sim_chunk)
        else:
            similarities = sim_func(casebase, query)

        ranking = _similarities2ranking(similarities)

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

        return {key: similarities[key] for key in ranking[: self.limit]}


def load(
    import_names: Sequence[str] | str,
) -> list[RetrieverFunc[Any, Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: list[RetrieverFunc[Any, Any, Any]] = []

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Sequence):
            assert all(isinstance(func, Callable) for func in retrievers)
            retrievers.extend(obj)
        elif isinstance(obj, Callable):
            retrievers.append(cast(RetrieverFunc[Any, Any, Any], obj))

    return retrievers


def load_map(
    import_names: Collection[str] | str,
) -> dict[str, RetrieverFunc[Any, Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: dict[str, RetrieverFunc] = {}

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
