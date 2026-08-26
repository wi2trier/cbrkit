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
    GroupCasebase,
    MaybeFactory,
    RetrieverFunc,
    SimMap,
)

logger = get_logger(__name__)

__all__ = ["as_groups", "build"]


def as_groups[K, V](casebase: Casebase[K, V]) -> GroupCasebase[K, V]:
    """View a casebase as a casebase of single-case groups.

    Grouped retrieval represents a group of source cases as a single case, which
    lets a retriever score a relation over several cases at once, for instance
    whether a query lies between a pair of cases.
    This function is the entry point into that representation and
    `cbrkit.retrieval.group` then combines them into wider groups.
    Queries have to be wrapped in a single-case group as well.

    Args:
        casebase: Source casebase to view as groups.

    Returns:
        A casebase mapping each source key to a group holding only that case.

    Examples:
        >>> as_groups({"a": 1, "b": 2})
        {('a',): (1,), ('b',): (2,)}
    """
    return {(key,): (case,) for key, case in casebase.items()}


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](RetrieverFunc[K, V, S]):
    """Creates a retriever from a similarity function.

    This is the core building block for retrieval pipelines.
    It flattens all (casebase, query) batches into a single list of pairwise
    comparisons and evaluates them together using the similarity function.
    When multiprocessing is enabled, parallelism occurs at the level of
    individual similarity computations within batches.

    To parallelize across batches instead (i.e., process each (casebase, query)
    pair independently), wrap the result with `distribute`.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        multiprocessing: Either a boolean to enable multiprocessing with all cores
            or an integer to specify the number of processes to use or a multiprocessing.Pool object.
            Parallelizes the similarity computations within batches.
        chunksize: Number of pairs to process at a time using the similarity function.
            If 0, it will be set to the number of pairs divided by the number of processes.

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
