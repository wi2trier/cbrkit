from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import Any, cast, override

from .helpers import (
    SimPairWrapper,
    get_metadata,
    similarities2ranking,
    unpack_sim,
)
from .typing import (
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

__all__ = [
    "dropout",
    "discard",
    "build",
    "apply_queries",
    "apply_query",
    "apply_single",
    "Result",
    "ResultStep",
    "QueryResultStep",
]


@dataclass(slots=True, frozen=True)
class QueryResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    casebase: Casebase[K, V]

    @property
    def similarity(self) -> S:
        if len(self.similarities) != 1:
            raise ValueError("The step contains multiple similarities.")

        return next(iter(self.similarities.values()))

    @property
    def case(self) -> V:
        if len(self.casebase) != 1:
            raise ValueError("The step contains multiple cases.")

        return next(iter(self.casebase.values()))

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]

        return x


@dataclass(slots=True, frozen=True)
class ResultStep[Q, C, V, S: Float]:
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonDict

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        if len(self.queries) != 1:
            raise ValueError("The step contains multiple queries.")

        return next(iter(self.queries.values()))

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.default_query.similarities

    @property
    def casebase(self) -> Mapping[C, V]:
        return self.default_query.casebase

    @property
    def similarity(self) -> S:
        return self.default_query.similarity

    @property
    def case(self) -> V:
        return self.default_query.case


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    steps: list[ResultStep[Q, C, V, S]]

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        return self.steps[-1]

    @property
    def metadata(self) -> JsonDict:
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        return self.final_step.queries

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.similarities

    @property
    def casebase(self) -> Mapping[C, V]:
        return self.final_step.casebase

    @property
    def similarity(self) -> S:
        return self.final_step.similarity

    @property
    def case(self) -> V:
        return self.final_step.case

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for step in x["steps"]:
            for item in step["queries"].values():
                del item["casebase"]

        return x


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


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
    similarities: Mapping[C, S] | None = None,
) -> Result[Q, C, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase for the query.
        queries: The queries that will be used to adapt the casebase.
        reusers: The reuser functions that will be applied to the casebase.

    Returns:
        Returns an object of type Result
    """

    if not isinstance(reusers, Sequence):
        reusers = [reusers]

    steps: list[ResultStep[Q, C, V, S]] = []
    current_casebases: Mapping[Q, Mapping[C, V]] = {
        query_key: casebase for query_key in queries
    }
    current_similarities: Mapping[Q, Mapping[C, S] | None] = {
        query_key: similarities for query_key in queries
    }

    for reuser in reusers:
        queries_results = reuser(
            [
                (current_casebases[query_key], query, current_similarities[query_key])
                for query_key, query in queries.items()
            ]
        )
        step_queries = {
            query_key: QueryResultStep(
                adapted_sims,
                adapted_casebase,
            )
            for query_key, (adapted_casebase, adapted_sims) in zip(
                queries, queries_results, strict=True
            )
        }

        step = ResultStep(step_queries, get_metadata(reuser))
        steps.append(step)
        current_casebases = {
            query_key: step_queries[query_key].casebase for query_key in queries
        }
        current_similarities = {
            query_key: step_queries[query_key].similarities for query_key in queries
        }

    return Result(steps)


def apply_single[V, S: Float](
    case: V,
    query: V,
    reusers: ReuserFunc[str, V, S] | Sequence[ReuserFunc[str, V, S]],
    similarity: S | None = None,
) -> Result[str, str, V, S]:
    """Applies a single query to a single case using reuser functions.

    Args:
        case: The case that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(
        {"default": case},
        {"default": query},
        reusers,
        {"default": similarity} if similarity is not None else None,
    )


def apply_query[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[K, V, S] | Sequence[ReuserFunc[K, V, S]],
    similarities: Mapping[K, S] | None = None,
) -> Result[str, K, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, reusers, similarities)


apply = apply_query
