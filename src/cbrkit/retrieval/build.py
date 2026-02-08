import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import override

from ..helpers import (
    batchify_sim,
    chunkify,
    get_logger,
    mp_count,
    mp_map,
    use_mp,
)
from ..typing import (
    AnySimFunc,
    Casebase,
    Float,
    MaybeFactory,
    RetrieverFunc,
    SimMap,
)

logger = get_logger(__name__)


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
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
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

        return [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(batches, similarities, strict=True)
        ]
