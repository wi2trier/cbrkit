from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import override

from ..helpers import batchify_sim, get_logger, mp_starmap, produce_factory
from ..typing import (
    AdaptationFunc,
    AnySimFunc,
    Casebase,
    Float,
    MaybeFactory,
    ReviserFunc,
    SimMap,
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

    Returns:
        The revised casebases with quality scores.

    Examples:
        >>> import cbrkit
        >>> reviser = build(
        ...     assess_func=cbrkit.sim.generic.equality(),
        ... )
    """

    assess_func: MaybeFactory[AnySimFunc[V, S]]
    repair_func: MaybeFactory[AdaptationFunc[V]] | None = None
    multiprocessing: Pool | int | bool = False

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
        repair_func: AdaptationFunc[V],
    ) -> Sequence[Casebase[K, V]]:
        batches_index: list[tuple[int, K]] = []
        flat_batches: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(batches):
            for key, case in casebase.items():
                batches_index.append((idx, key))
                flat_batches.append((case, query))

        repaired_cases = mp_starmap(
            repair_func,
            flat_batches,
            self.multiprocessing,
            logger,
        )
        repaired_casebases: list[dict[K, V]] = [{} for _ in range(len(batches))]

        for (idx, key), repaired_case in zip(
            batches_index, repaired_cases, strict=True
        ):
            repaired_casebases[idx][key] = repaired_case

        return repaired_casebases
