import math
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import override

from ..helpers import (
    batchify_sim,
    chunkify,
    get_logger,
    get_value,
    mp_count,
    mp_map,
    sim_map2ranking,
    unpack_float,
    use_mp,
)
from ..typing import (
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

    Args:
        conversion_func: A function that converts the input values from one type to another.
        retriever_func: The retriever function to be used on the converted values.
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
class build[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Based on the similarity function this function creates a retriever function.

    The given limit will be applied after filtering for min/max similarity.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        processes: Number of processes to use. If processes is less than 1, the number returned by os.cpu_count() is used.
        similarity_chunksize: Number of batches to process in each chunk.

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
        ...                 }
        ...             ),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     )
        ... )
    """

    similarity_func: MaybeFactory[AnySimFunc[V, S]]
    multiprocessing: Pool | int | bool = False
    chunksize: int | None = None

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]:
        sim_func = batchify_sim(self.similarity_func)
        similarities: list[dict[K, S]] = []

        flat_sims: Sequence[S] = []
        flat_batches_index: list[tuple[int, K]] = []
        flat_batches: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(batches):
            similarities.append({})

            for key, case in casebase.items():
                flat_batches_index.append((idx, key))
                flat_batches.append((case, query))

        if use_mp(self.multiprocessing):
            chunksize = self.chunksize or math.ceil(
                len(flat_batches) / mp_count(self.multiprocessing)
            )
            pair_chunks = list(chunkify(flat_batches, chunksize))
            sim_chunks = mp_map(sim_func, pair_chunks, self.multiprocessing, logger)

            for sim_chunk in sim_chunks:
                flat_sims.extend(sim_chunk)

        else:
            flat_sims = sim_func(flat_batches)

        for (idx, key), sim in zip(flat_batches_index, flat_sims, strict=True):
            similarities[idx][key] = sim

        return similarities
