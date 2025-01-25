from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature as inspect_signature
from multiprocessing.pool import Pool
from typing import cast, override

from ..helpers import get_logger, mp_starmap, produce_factory
from ..typing import (
    AdaptationFunc,
    AnyAdaptationFunc,
    Casebase,
    Float,
    MapAdaptationFunc,
    MaybeFactory,
    ReduceAdaptationFunc,
    RetrieverFunc,
    ReuserFunc,
    SimMap,
)

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](ReuserFunc[K, V, S]):
    """Builds a casebase by adapting cases using an adaptation function and a similarity function.

    Args:
        adaptation_func: The adaptation function that will be applied to the cases.
        retriever_func: The similarity function that will be used to compare the adapted cases to the query.
        processes: The number of processes that will be used to apply the adaptation function. If processes is set to 1, the adaptation function will be applied in the main process.

    Returns:
        The adapted casebases and the similarities between the adapted cases and the query.
    """

    adaptation_func: MaybeFactory[AnyAdaptationFunc[K, V]]
    retriever_func: RetrieverFunc[K, V, S]
    multiprocessing: Pool | int | bool = False

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        adaptation_func = produce_factory(self.adaptation_func)
        adapted_casebases = self._adapt(batches, adaptation_func)
        adapted_batches = [
            (adapted_casebase, query)
            for adapted_casebase, (_, query) in zip(
                adapted_casebases, batches, strict=True
            )
        ]
        adapted_similarities = self.retriever_func(adapted_batches)

        return [
            (adapted_casebase, adapted_sim)
            for adapted_casebase, adapted_sim in zip(
                adapted_casebases, adapted_similarities, strict=True
            )
        ]

    def _adapt(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        adaptation_func: AnyAdaptationFunc[K, V],
    ) -> Sequence[Casebase[K, V]]:
        adaptation_func_signature = inspect_signature(adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adapt_func = cast(
                MapAdaptationFunc[K, V] | ReduceAdaptationFunc[K, V],
                adaptation_func,
            )
            adaptation_results = mp_starmap(
                adapt_func,
                batches,
                self.multiprocessing,
                logger,
            )

            if all(isinstance(item, tuple) for item in adaptation_results):
                adaptation_results = cast(Sequence[tuple[K, V]], adaptation_results)
                return [
                    {adapted_key: adapted_case}
                    for adapted_key, adapted_case in adaptation_results
                ]

            return cast(Sequence[Casebase[K, V]], adaptation_results)

        adapt_func = cast(AdaptationFunc[V], adaptation_func)
        batches_index: list[tuple[int, K]] = []
        flat_batches: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(batches):
            for key, case in casebase.items():
                batches_index.append((idx, key))
                flat_batches.append((case, query))

        adapted_cases = mp_starmap(
            adapt_func,
            flat_batches,
            self.multiprocessing,
            logger,
        )
        adapted_casebases: list[dict[K, V]] = [{} for _ in range(len(batches))]

        for (idx, key), adapted_case in zip(batches_index, adapted_cases, strict=True):
            adapted_casebases[idx][key] = adapted_case

        return adapted_casebases
