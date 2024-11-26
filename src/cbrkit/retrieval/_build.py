from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from typing import override

from ..helpers import (
    SimSeqWrapper,
    get_metadata,
    similarities2ranking,
    unpack_sim,
)
from ..typing import (
    AnySimFunc,
    Casebase,
    Float,
    JsonDict,
    RetrieverFunc,
    SimMap,
    SupportsMetadata,
)


def chunkify[V](val: Sequence[V], k: int) -> Iterator[Sequence[V]]:
    """Yield a total of k chunks from val.

    Examples:
        >>> list(chunkify([1, 2, 3, 4, 5, 6, 7, 8, 9], 4))
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """

    for i in range(0, len(val), k):
        yield val[i : i + k]


@dataclass(slots=True, frozen=True)
class dropout[K, V, S: Float](RetrieverFunc[K, V, S], SupportsMetadata):
    retriever_func: RetrieverFunc[K, V, S]
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "retriever_func": get_metadata(self.retriever_func),
            "limit": self.limit,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }

    @override
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        return [self._filter(entry) for entry in self.retriever_func(pairs)]

    def _filter(
        self,
        similarities: SimMap[K, S],
    ) -> dict[K, S]:
        ranking: list[K] = similarities2ranking(similarities)

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

        return {key: similarities[key] for key in ranking}


@dataclass(slots=True, frozen=True)
class transpose[K, U, V, S: Float](RetrieverFunc[K, V, S], SupportsMetadata):
    """Transforms a retriever function from one type to another.

    Args:
        conversion_func: A function that converts the input values from one type to another.
        retriever_func: The retriever function to be used on the converted values.
    """

    conversion_func: Callable[[V], U]
    retriever_func: RetrieverFunc[K, U, S]

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "conversion_func": get_metadata(self.conversion_func),
            "retriever_func": get_metadata(self.retriever_func),
        }

    @override
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
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
                for casebase, query in pairs
            ]
        )


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](RetrieverFunc[K, V, S], SupportsMetadata):
    """Based on the similarity function this function creates a retriever function.

    The given limit will be applied after filtering for min/max similarity.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        processes: Number of processes to use. If processes is less than 1, the number returned by os.cpu_count() is used.
        similarity_chunksize: Number of pairs to process in each chunk.

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
        ...     )
        ... )
    """

    similarity_func: AnySimFunc[V, S]
    processes: int = 1
    similarity_chunksize: int = 1

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "similarity_func": get_metadata(self.similarity_func),
            "processes": self.processes,
            "similarity_chunksize": self.similarity_chunksize,
        }

    @override
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        sim_func = SimSeqWrapper(self.similarity_func)
        similarities: list[dict[K, S]] = []

        flat_sims: Sequence[S] = []
        flat_pairs_index: list[tuple[int, K]] = []
        flat_pairs: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(pairs):
            similarities.append({})

            for key, case in casebase.items():
                flat_pairs_index.append((idx, key))
                flat_pairs.append((case, query))

        if self.processes != 1:
            pool_processes = None if self.processes <= 0 else self.processes
            pair_chunks = chunkify(flat_pairs, self.similarity_chunksize)

            with Pool(pool_processes) as pool:
                sim_chunks = pool.map(sim_func, pair_chunks)

            for sim_chunk in sim_chunks:
                flat_sims.extend(sim_chunk)
        else:
            flat_sims = sim_func(flat_pairs)

        for (idx, key), sim in zip(flat_pairs_index, flat_sims, strict=True):
            similarities[idx][key] = sim

        return similarities
