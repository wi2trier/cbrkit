import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any, Literal, override

from ..helpers import (
    batchify_sim,
    chunkify,
    get_logger,
    get_value,
    mp_count,
    mp_map,
    mp_starmap,
    sim_map2ranking,
    unpack_float,
    use_mp,
)
from ..sim.aggregator import default_aggregator
from ..typing import (
    AggregatorFunc,
    AnySimFunc,
    Casebase,
    ConversionFunc,
    Float,
    MaybeFactory,
    RetrieverFunc,
    SimMap,
    StructuredValue,
)

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class dropout[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Filters the retrieved cases based on the similarity values.

    Args:
        retriever_func: The retriever function to be used.
            Typically constructed with the `build` function.
        limit: The maximum number of cases to be returned.
        min_similarity: The minimum similarity value to be considered.
        max_similarity: The maximum similarity value to be considered.

    Returns:
        A retriever function that filters the retrieved cases based on the similarity values.
    """

    retriever_func: RetrieverFunc[K, V, S]
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        return [self._call_single(entry) for entry in self.retriever_func(batches)]

    def _call_single(self, similarities: SimMap[K, S]) -> SimMap[K, S]:
        ranking = sim_map2ranking(similarities)

        if self.min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_float(similarities[key]) >= self.min_similarity
            ]
        if self.max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_float(similarities[key]) <= self.max_similarity
            ]
        if self.limit is not None:
            ranking = ranking[: self.limit]

        return {key: similarities[key] for key in ranking}


@dataclass(slots=True, frozen=True)
class transpose[K, V1, V2, S: Float](RetrieverFunc[K, V1, S]):
    """Transforms a retriever function from one type to another.

    Useful when the input values need to be converted before retrieval,
    for instance, when the cases are nested and you only need to compare a subset of the values.

    Args:
        conversion_func: A function that converts the input values from one type to another.
        retriever_func: The retriever function to be used on the converted values.

    Returns:
        A retriever function that works on the converted values
    """

    retriever_func: RetrieverFunc[K, V2, S]
    conversion_func: ConversionFunc[V1, V2]

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V1], V1]]
    ) -> Sequence[SimMap[K, S]]:
        return self.retriever_func(
            [
                (
                    {
                        key: self.conversion_func(value)
                        for key, value in casebase.items()
                    },
                    self.conversion_func(query),
                )
                for casebase, query in batches
            ]
        )


def transpose_value[K, V, S: Float](
    retriever_func: RetrieverFunc[K, V, S],
) -> RetrieverFunc[K, StructuredValue[V], S]:
    return transpose(retriever_func, get_value)


@dataclass(slots=True, frozen=True)
class combine[K, V, S: Float](RetrieverFunc[K, V, float]):
    """Combines multiple retriever functions into one.

    Args:
        retriever_funcs: A list of retriever functions to be combined.
        aggregator: A function to aggregate the results from the retriever functions.
        strategy: The strategy to combine the results. Either "intersection" or "union".

    Returns:
        A retriever function that combines the results from multiple retrievers.
    """

    retriever_funcs: list[RetrieverFunc[K, V, S]]
    aggregator: AggregatorFunc[Any, S] = default_aggregator
    strategy: Literal["intersection", "union"] = "union"

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, float]]:
        results = [retriever_func(batches) for retriever_func in self.retriever_funcs]

        return [
            self.__call_batch__(
                [
                    results[retriever_idx][batch_idx]
                    for retriever_idx in range(len(self.retriever_funcs))
                ]
            )
            for batch_idx in range(len(batches))
        ]

    def __call_batch__(self, results: list[SimMap[K, S]]) -> SimMap[K, float]:
        if self.strategy == "intersection":
            return {
                key: self.aggregator(
                    [result[key] for result in results if key in result]
                )
                for key in set().intersection(
                    *[set(result.keys()) for result in results]
                )
            }

        elif self.strategy == "union":
            return {
                key: self.aggregator(
                    [result[key] for result in results if key in result]
                )
                for result in results
                for key in result.keys()
            }

        raise ValueError(f"Unknown strategy: {self.strategy}")


@dataclass(slots=True, frozen=True)
class distribute[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Distributes the retrieval process by passing each batch separately to the retriever function.

    Args:
        retriever_func: The retriever function to be used.
            Typically constructed with the `build` function.
        multiprocessing: Either a boolean to enable multiprocessing with all cores
            or an integer to specify the number of processes to use or a multiprocessing.Pool object.

    Returns:
        A retriever function that distributes the retrieval process.
    """

    retriever_func: RetrieverFunc[K, V, S]
    multiprocessing: Pool | int | bool

    def __call_batch__(self, x: Casebase[K, V], y: V) -> SimMap[K, S]:
        return self.retriever_func([(x, y)])[0]

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        return mp_starmap(self.__call_batch__, batches, self.multiprocessing, logger)


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Based on the similarity function this function creates a retriever function.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        multiprocessing: Either a boolean to enable multiprocessing with all cores
            or an integer to specify the number of processes to use or a multiprocessing.Pool object.
        chunksize: Number of batches to process at a time using the similarity function.
            If None, it will be set to the number of batches divided by the number of processes.

    Returns:
        A retriever function that computes the similarity between cases.

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
        ...                 }
        ...             ),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     )
        ... )
    """

    similarity_func: MaybeFactory[AnySimFunc[V, S]]
    multiprocessing: Pool | int | bool = False
    chunksize: int = 0

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        sim_func = batchify_sim(self.similarity_func)
        similarities: list[dict[K, S]] = [{} for _ in range(len(batches))]

        flat_sims: Sequence[S] = []
        flat_batches_index: list[tuple[int, K]] = []
        flat_batches: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(batches):
            for key, case in casebase.items():
                flat_batches_index.append((idx, key))
                flat_batches.append((case, query))

        if use_mp(self.multiprocessing) or self.chunksize > 0:
            chunksize = (
                self.chunksize
                if self.chunksize > 0
                else len(flat_batches) // mp_count(self.multiprocessing)
            )
            batch_chunks = list(chunkify(flat_batches, chunksize))
            sim_chunks = mp_map(sim_func, batch_chunks, self.multiprocessing, logger)
            flat_sims = list(itertools.chain.from_iterable(sim_chunks))

        else:
            flat_sims = sim_func(flat_batches)

        for (idx, key), sim in zip(flat_batches_index, flat_sims, strict=True):
            similarities[idx][key] = sim

        return similarities
