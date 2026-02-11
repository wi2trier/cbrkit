import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import override

from ..helpers import (
    batchify_adaptation,
    batchify_sim,
    chunkify,
    get_logger,
    mp_count,
    mp_map,
    produce_factory,
    use_mp,
)
from ..typing import (
    AnySimFunc,
    Casebase,
    Float,
    MaybeFactory,
    ReviserFunc,
    SimMap,
    SimpleAdaptationFunc,
)

logger = get_logger(__name__)

__all__ = ["build"]


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](ReviserFunc[K, V, S]):
    """Builds a reviser that assesses solution quality and optionally repairs solutions.

    The reviser first optionally repairs solutions using the repair function,
    then assesses their quality using the assess function (a similarity function
    comparing solutions to queries).

    Args:
        assess_func: Similarity function to evaluate solution quality.
        repair_func: Optional adaptation function to repair solutions before assessment.
        multiprocessing: Multiprocessing configuration for repair.
        chunksize: Number of batches to process at a time using the repair function.
            If 0, it will be set to the number of batches divided by the number of processes.

    Returns:
        The revised casebases with quality scores.

    Examples:
        >>> import cbrkit
        >>> reviser = build(
        ...     assess_func=cbrkit.sim.generic.equality(),
        ... )
    """

    assess_func: MaybeFactory[AnySimFunc[V, S]]
    repair_func: MaybeFactory[SimpleAdaptationFunc[V]] | None = None
    multiprocessing: Pool | int | bool = False
    chunksize: int = 0

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        current_batches = batches

        # Step 1: Optionally repair solutions
        if self.repair_func is not None:
            repair_func = produce_factory(self.repair_func)
            repaired_casebases = self._repair(current_batches, repair_func)
            current_batches = [
                (repaired_casebase, query)
                for repaired_casebase, (_, query) in zip(
                    repaired_casebases, current_batches, strict=True
                )
            ]

        # Step 2: Assess quality of all solutions
        assess_func = batchify_sim(produce_factory(self.assess_func))

        flat_batches: list[tuple[V, V]] = []
        flat_index: list[tuple[int, K]] = []

        for idx, (casebase, query) in enumerate(current_batches):
            for key, solution in casebase.items():
                flat_index.append((idx, key))
                flat_batches.append((solution, query))

        quality_scores = assess_func(flat_batches)

        quality_maps: list[dict[K, S]] = [{} for _ in current_batches]
        for (idx, key), quality in zip(flat_index, quality_scores, strict=True):
            quality_maps[idx][key] = quality

        return [
            (casebase, quality_map)
            for (casebase, _), quality_map in zip(
                current_batches, quality_maps, strict=True
            )
        ]

    def _repair(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        repair_func: SimpleAdaptationFunc[V],
    ) -> Sequence[Casebase[K, V]]:
        batch_repair_func = batchify_adaptation(repair_func)
        batches_index: list[tuple[int, K]] = []
        flat_batches: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(batches):
            for key, case in casebase.items():
                batches_index.append((idx, key))
                flat_batches.append((case, query))

        repaired_cases: Sequence[V]

        if use_mp(self.multiprocessing) or self.chunksize > 0:
            chunksize = (
                self.chunksize
                if self.chunksize > 0
                else len(flat_batches) // mp_count(self.multiprocessing)
            )
            batch_chunks = list(chunkify(flat_batches, chunksize))
            repaired_chunks = mp_map(
                batch_repair_func, batch_chunks, self.multiprocessing, logger
            )
            repaired_cases = list(itertools.chain.from_iterable(repaired_chunks))
        else:
            repaired_cases = list(batch_repair_func(flat_batches))

        repaired_casebases: list[dict[K, V]] = [{} for _ in range(len(batches))]

        for (idx, key), repaired_case in zip(
            batches_index, repaired_cases, strict=True
        ):
            repaired_casebases[idx][key] = repaired_case

        return repaired_casebases
