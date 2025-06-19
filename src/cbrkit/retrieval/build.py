import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Literal, override

from ..helpers import (
    batchify_sim,
    chunkify,
    get_logger,
    get_value,
    mp_count,
    mp_map,
    mp_starmap,
    optional_dependencies,
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
        default_sim: The default similarity value to use for strategy "union" when a case is not found in one of the retriever results.

    Returns:
        A retriever function that combines the results from multiple retrievers.
    """

    retriever_funcs: (
        Sequence[RetrieverFunc[K, V, S]] | Mapping[str, RetrieverFunc[K, V, S]]
    )
    aggregator: AggregatorFunc[str, S | float] = default_aggregator
    strategy: Literal["intersection", "union"] = "union"
    default_sim: float = 0.0

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, float]]:
        if isinstance(self.retriever_funcs, Sequence):
            func_results = [
                retriever_func(batches) for retriever_func in self.retriever_funcs
            ]

            return [
                self.__call_batch__(
                    [batch_results[batch_idx] for batch_results in func_results]
                )
                for batch_idx in range(len(batches))
            ]

        elif isinstance(self.retriever_funcs, Mapping):
            results = {
                func_key: retriever_func(batches)
                for func_key, retriever_func in self.retriever_funcs.items()
            }

            return [
                self.__call_batch__(
                    {
                        func_key: func_results[batch_idx]
                        for func_key, func_results in results.items()
                    }
                )
                for batch_idx in range(len(batches))
            ]

        raise ValueError(f"Invalid retriever_funcs type: {type(self.retriever_funcs)}")

    def __call_batch__(
        self, results: Sequence[SimMap[K, S]] | Mapping[str, SimMap[K, S]]
    ) -> SimMap[K, float]:
        case_keys: set[K]

        if isinstance(results, Sequence):
            if self.strategy == "intersection":
                case_keys = set().intersection(*(result.keys() for result in results))
            elif self.strategy == "union":
                case_keys = set().union(*(result.keys() for result in results))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            return {
                case_key: self.aggregator(
                    [result.get(case_key, self.default_sim) for result in results]
                )
                for case_key in case_keys
            }

        elif isinstance(results, Mapping):
            if self.strategy == "intersection":
                case_keys = set().intersection(
                    *(result.keys() for result in results.values())
                )
            elif self.strategy == "union":
                case_keys = set().union(*(result.keys() for result in results.values()))
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            return {
                case_key: self.aggregator(
                    {
                        func_key: result.get(case_key, self.default_sim)
                        for func_key, result in results.items()
                    }
                )
                for case_key in case_keys
            }

        raise ValueError(f"Invalid results type: {type(results)}")


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


with optional_dependencies():
    from chonkie import BaseChunker

    @dataclass(slots=True, frozen=True)
    class chunk[S: Float](RetrieverFunc[str, str, S]):
        """Chunks string cases using the chonkie library before retrieval.

        This retriever is special in that it returns a different set of cases for each batch
        it processes, as it splits the original string cases into chunks.

        Args:
            retriever_func: The retriever function to be used on the chunked strings.
            chunker: A BaseChunker instance from the chonkie library.

        Returns:
            A retriever function that chunks string cases and retrieves from the chunks.
        """

        retriever_func: RetrieverFunc[str, str, S]
        chunker: BaseChunker

        @override
        def __call__(
            self, batches: Sequence[tuple[Casebase[str, str], str]]
        ) -> Sequence[SimMap[str, S]]:
            chunked_batches: list[tuple[Casebase[str, str], str]] = []

            for casebase, query in batches:
                chunked_casebase: dict[str, str] = {}

                for case_key, case_text in casebase.items():
                    chunks = self.chunker.chunk(case_text)

                    for i, chunk in enumerate(chunks):
                        chunk_key = f"{case_key}-chunk{i}"
                        chunked_casebase[chunk_key] = (
                            chunk if isinstance(chunk, str) else chunk.text
                        )

                chunked_batches.append((chunked_casebase, query))

            return self.retriever_func(chunked_batches)
