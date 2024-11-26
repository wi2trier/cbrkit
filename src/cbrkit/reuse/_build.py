from collections.abc import Sequence
from dataclasses import dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import cast, override

from ..helpers import (
    SimPairWrapper,
    get_metadata,
    similarities2ranking,
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
    JsonDict,
    ReuserFunc,
    SimMap,
    SupportsMetadata,
)


@dataclass(slots=True, frozen=True, kw_only=True)
class discard[K, V, S: Float](ReuserFunc[K, V, S], SupportsMetadata):
    reuser_func: ReuserFunc[K, V, S]
    similarity_delta: float

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "reuser_func": get_metadata(self.reuser_func),
            "similarity_delta": self.similarity_delta,
        }

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        return [
            self._filter(adapted_casebase, adapted_sims, prev_casebase, prev_sims)
            for (adapted_casebase, adapted_sims), (
                prev_casebase,
                _,
                prev_sims,
            ) in zip(self.reuser_func(pairs), pairs, strict=True)
        ]

    def _filter(
        self,
        adapted_casebase: Casebase[K, V],
        adapted_sims: SimMap[K, S],
        prev_casebase: Casebase[K, V],
        prev_sims: SimMap[K, S] | None,
    ) -> tuple[Casebase[K, V], SimMap[K, S]]:
        if prev_sims is None:
            raise ValueError("similarity_delta requires existing similarities")

        new_casebase: dict[K, V] = {}
        new_sims: dict[K, S] = {}

        for key in adapted_casebase:
            if (
                unpack_sim(adapted_sims[key])
                >= unpack_sim(prev_sims[key]) + self.similarity_delta
            ):
                new_casebase[key] = adapted_casebase[key]
                new_sims[key] = adapted_sims[key]
            else:
                new_casebase[key] = prev_casebase[key]
                new_sims[key] = prev_sims[key]

        return new_casebase, new_sims


@dataclass(slots=True, frozen=True, kw_only=True)
class dropout[K, V, S: Float](ReuserFunc[K, V, S], SupportsMetadata):
    reuser_func: ReuserFunc[K, V, S]
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "reuser_func": get_metadata(self.reuser_func),
            "limit": self.limit,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }

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
        ranking: list[K] = similarities2ranking(adapted_sims)

        if self.min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(adapted_sims[key]) >= self.min_similarity
            ]
        if self.max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(adapted_sims[key]) <= self.max_similarity
            ]
        if self.limit is not None:
            ranking = ranking[: self.limit]

        return {key: adapted_casebase[key] for key in ranking}, {
            key: adapted_sims[key] for key in ranking
        }


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](ReuserFunc[K, V, S], SupportsMetadata):
    """Builds a casebase by adapting cases using an adaptation function and a similarity function.

    Args:
        adaptation_func: The adaptation function that will be applied to the cases.
        similarity_func: The similarity function that will be used to compare the adapted cases to the query.
        processes: The number of processes that will be used to apply the adaptation function. If processes is set to 1, the adaptation function will be applied in the main process.

    Returns:
        The adapted casebase.
    """

    adaptation_func: AnyAdaptFunc[K, V]
    similarity_func: AnySimFunc[V, S]
    processes: int = 1

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "adaptation_func": get_metadata(self.adaptation_func),
            "similarity_func": get_metadata(self.similarity_func),
            "processes": self.processes,
        }

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        sim_func = SimPairWrapper(self.similarity_func)
        adapted_casebases: list[tuple[dict[K, V], dict[K, S]]] = []

        adaptation_func_signature = inspect_signature(self.adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adaptation_func = cast(
                AdaptMapFunc[K, V] | AdaptReduceFunc[K, V],
                self.adaptation_func,
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
                    adaptation_func(casebase, query) for casebase, query, _ in pairs
                ]

            if all(isinstance(item, tuple) for item in adaptation_results):
                adaptation_results = cast(Sequence[tuple[K, V]], adaptation_results)
                adapted_casebases = [
                    (
                        {adapted_key: adapted_case},
                        {adapted_key: sim_func(adapted_case, query)},
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
                            key: sim_func(adapted_case, query)
                            for key, adapted_case in adapted_casebase.items()
                        },
                    )
                    for (_, query, _), adapted_casebase in zip(
                        pairs, adaptation_results, strict=True
                    )
                ]

        else:
            adaptation_func = cast(AdaptPairFunc[V], self.adaptation_func)
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
                        adaptation_func,
                        flat_pairs,
                    )
                    adapted_sims = pool.starmap(
                        sim_func,
                        (
                            (adapted_case, query)
                            for (_, query), adapted_case in zip(
                                flat_pairs, adapted_cases, strict=True
                            )
                        ),
                    )
            else:
                adapted_cases = [
                    adaptation_func(case, query) for case, query in flat_pairs
                ]
                adapted_sims = [
                    sim_func(adapted_case, query)
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
