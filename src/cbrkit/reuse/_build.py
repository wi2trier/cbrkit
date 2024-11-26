from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import cast, override

from ..helpers import (
    SimPairWrapper,
    sim_dropout,
    unpack_sim,
)
from ..typing import (
    AdaptMapFunc,
    AdaptPairFunc,
    AdaptReduceFunc,
    AnyAdaptFunc,
    AnySimFunc,
    Casebase,
    Float,
    ReuserFunc,
    SimMap,
    SimPairFunc,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class dropout[K, V, S: Float](ReuserFunc[K, V, S]):
    reuser_func: ReuserFunc[K, V, S]
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        return [
            self._filter(adapted_casebase, adapted_sims)
            for (adapted_casebase, adapted_sims) in self.reuser_func(pairs)
        ]

    def _filter(
        self,
        adapted_casebase: Casebase[K, V],
        adapted_sims: SimMap[K, S],
    ) -> tuple[Casebase[K, V], SimMap[K, S]]:
        filtered_sims = sim_dropout(
            adapted_sims, self.limit, self.min_similarity, self.max_similarity
        )

        return {key: adapted_casebase[key] for key in filtered_sims}, filtered_sims


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](ReuserFunc[K, V, S]):
    """Builds a casebase by adapting cases using an adaptation function and a similarity function.

    Args:
        adaptation_func: The adaptation function that will be applied to the cases.
        similarity_func: The similarity function that will be used to compare the adapted cases to the query.
        similarity_delta: The allowed difference between the previous and the next similarity value.
        processes: The number of processes that will be used to apply the adaptation function. If processes is set to 1, the adaptation function will be applied in the main process.

    Returns:
        The adapted casebase.
    """

    adaptation_funcs: AnyAdaptFunc[K, V] | Sequence[AnyAdaptFunc[K, V]]
    similarity_func: AnySimFunc[V, S]
    similarity_delta: float = -1.0
    processes: int = 1

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        similarity_func = SimPairWrapper(self.similarity_func)
        adaptation_funcs = (
            self.adaptation_funcs
            if isinstance(self.adaptation_funcs, Sequence)
            else [self.adaptation_funcs]
        )
        current_pairs = pairs

        for adaptation_func in adaptation_funcs:
            adapted_casebases = self._adapt(
                current_pairs, adaptation_func, similarity_func
            )
            current_pairs = [
                self._filter(adapted_pair, current_pair)
                for adapted_pair, current_pair in zip(
                    adapted_casebases, current_pairs, strict=True
                )
            ]

        return [
            (
                casebase,
                sims
                if sims is not None
                else {
                    key: similarity_func(case, query) for key, case in casebase.items()
                },
            )
            for casebase, query, sims in current_pairs
        ]

    def _adapt(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
        adaptation_func: AnyAdaptFunc[K, V],
        similarity_func: SimPairFunc[V, S],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        adapted_casebases: list[tuple[dict[K, V], dict[K, S]]] = []

        adaptation_func_signature = inspect_signature(adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adapt_func = cast(
                AdaptMapFunc[K, V] | AdaptReduceFunc[K, V],
                adaptation_func,
            )

            if self.processes != 1:
                pool_processes = None if self.processes <= 0 else self.processes

                with Pool(pool_processes) as pool:
                    adaptation_results = pool.starmap(
                        adaptation_func,
                        ((casebase, query) for casebase, query, _ in pairs),
                    )
            else:
                adaptation_results = [
                    adapt_func(casebase, query) for casebase, query, _ in pairs
                ]

            if all(isinstance(item, tuple) for item in adaptation_results):
                adaptation_results = cast(Sequence[tuple[K, V]], adaptation_results)
                adapted_casebases = [
                    (
                        {adapted_key: adapted_case},
                        {adapted_key: similarity_func(adapted_case, query)},
                    )
                    for (_, query, _), (adapted_key, adapted_case) in zip(
                        pairs, adaptation_results, strict=True
                    )
                ]
            else:
                adaptation_results = cast(Sequence[Casebase[K, V]], adaptation_results)
                adapted_casebases = [
                    (
                        dict(adapted_casebase),
                        {
                            key: similarity_func(adapted_case, query)
                            for key, adapted_case in adapted_casebase.items()
                        },
                    )
                    for (_, query, _), adapted_casebase in zip(
                        pairs, adaptation_results, strict=True
                    )
                ]

        else:
            adapt_func = cast(AdaptPairFunc[V], adaptation_func)
            pairs_index: list[tuple[int, K]] = []
            flat_pairs: list[tuple[V, V]] = []

            for idx, (casebase, query, _) in enumerate(pairs):
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
                    adapted_sims = pool.starmap(
                        similarity_func,
                        (
                            (adapted_case, query)
                            for (_, query), adapted_case in zip(
                                flat_pairs, adapted_cases, strict=True
                            )
                        ),
                    )
            else:
                adapted_cases = [adapt_func(case, query) for case, query in flat_pairs]
                adapted_sims = [
                    similarity_func(adapted_case, query)
                    for (_, query), adapted_case in zip(
                        flat_pairs, adapted_cases, strict=True
                    )
                ]

            adapted_casebases = [({}, {}) for _ in range(len(pairs))]

            for (idx, key), adapted_case, adapted_sim in zip(
                pairs_index, adapted_cases, adapted_sims, strict=True
            ):
                adapted_casebases[idx][0][key] = adapted_case
                adapted_casebases[idx][1][key] = adapted_sim

        return adapted_casebases

    def _filter(
        self,
        adapted_pair: tuple[Casebase[K, V], SimMap[K, S]],
        current_pair: tuple[Casebase[K, V], V, SimMap[K, S] | None],
    ) -> tuple[Casebase[K, V], V, SimMap[K, S]]:
        current_casebase, query, current_sims = current_pair
        adapted_casebase, adapted_sims = adapted_pair

        if current_sims is None:
            raise ValueError("similarity_delta requires existing similarities")

        new_casebase: dict[K, V] = {}
        new_sims: dict[K, S] = {}

        for key in adapted_casebase:
            if (
                unpack_sim(adapted_sims[key])
                >= unpack_sim(current_sims[key]) + self.similarity_delta
            ):
                new_casebase[key] = adapted_casebase[key]
                new_sims[key] = adapted_sims[key]
            else:
                new_casebase[key] = current_casebase[key]
                new_sims[key] = current_sims[key]

        return new_casebase, query, new_sims
