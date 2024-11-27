from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import cast, override

from ..typing import (
    AdaptMapFunc,
    AdaptPairFunc,
    AdaptReduceFunc,
    AnyAdaptFunc,
    Casebase,
    Float,
    RetrieverFunc,
    ReuserFunc,
    SimMap,
)


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

    adaptation_func: AnyAdaptFunc[K, V]
    retriever_func: RetrieverFunc[K, V, S]
    processes: int = 1

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        adapted_casebases = self._adapt(pairs, self.adaptation_func)
        adapted_pairs = [
            (adapted_casebase, query)
            for adapted_casebase, (_, query) in zip(
                adapted_casebases, pairs, strict=True
            )
        ]
        adapted_similarities = self.retriever_func(adapted_pairs)

        return [
            (adapted_casebase, adapted_sim)
            for adapted_casebase, adapted_sim in zip(
                adapted_casebases, adapted_similarities, strict=True
            )
        ]

    def _adapt(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V]],
        adaptation_func: AnyAdaptFunc[K, V],
    ) -> Sequence[Casebase[K, V]]:
        adaptation_func_signature = inspect_signature(adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adapt_func = cast(
                AdaptMapFunc[K, V] | AdaptReduceFunc[K, V],
                adaptation_func,
            )

            if self.processes != 1:
                pool_processes = None if self.processes <= 0 else self.processes

                with Pool(pool_processes) as pool:
                    adaptation_results = pool.starmap(adaptation_func, pairs)
            else:
                adaptation_results = [
                    adapt_func(casebase, query) for casebase, query in pairs
                ]

            if all(isinstance(item, tuple) for item in adaptation_results):
                adaptation_results = cast(Sequence[tuple[K, V]], adaptation_results)
                return [
                    {adapted_key: adapted_case}
                    for adapted_key, adapted_case in adaptation_results
                ]

            return cast(Sequence[Casebase[K, V]], adaptation_results)

        adapt_func = cast(AdaptPairFunc[V], adaptation_func)
        pairs_index: list[tuple[int, K]] = []
        flat_pairs: list[tuple[V, V]] = []

        for idx, (casebase, query) in enumerate(pairs):
            for key, case in casebase.items():
                pairs_index.append((idx, key))
                flat_pairs.append((case, query))

        if self.processes != 1:
            pool_processes = None if self.processes <= 0 else self.processes

            with Pool(pool_processes) as pool:
                adapted_cases = pool.starmap(
                    adapt_func,
                    flat_pairs,
                )
        else:
            adapted_cases = [adapt_func(case, query) for case, query in flat_pairs]

        adapted_casebases: list[dict[K, V]] = [{} for _ in range(len(pairs))]

        for (idx, key), adapted_case in zip(pairs_index, adapted_cases, strict=True):
            adapted_casebases[idx][key] = adapted_case

        return adapted_casebases
