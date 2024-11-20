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

    @classmethod
    def build(
        cls,
        reuse_result: Casebase[K, tuple[V | None, S]],
        original_casebase: Casebase[K, V],
    ) -> "QueryResultStep[K, V, S]":
        casebase = {}
        sims = {}

        for key, (case, sim) in reuse_result.items():
            casebase[key] = case if case is not None else original_casebase[key]
            sims[key] = sim

        return cls(sims, casebase)

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]

        return x


@dataclass(slots=True, frozen=True)
class ResultStep[Q, C, V, S: Float]:
    by_query: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonDict

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        if len(self.by_query) != 1:
            raise ValueError("The step contains multiple queries.")

        return next(iter(self.by_query.values()))

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
    def by_query(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        return self.final_step.by_query

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
class base_reuser[K, V, S: Float](SupportsMetadata):
    limit: int | None = None
    min_similarity: float | None = None
    max_similarity: float | None = None

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            "limit": self.limit,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
        }

    def postprocess(
        self, casebase: Casebase[K, tuple[V | None, S]]
    ) -> Casebase[K, tuple[V | None, S]]:
        similarities = {key: sim for key, (_, sim) in casebase.items()}
        ranking = similarities2ranking(similarities)

        if self.min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) >= self.min_similarity
            ]
        if self.max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) <= self.max_similarity
            ]
        if self.limit is not None:
            ranking = ranking[: self.limit]

        return {key: casebase[key] for key in ranking}


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](base_reuser[K, V, S], ReuserFunc[K, V, S]):
    """Builds a casebase by adapting cases using an adaptation function and a similarity function.

    Args:
        adaptation_func: The adaptation function that will be applied to the cases.
        similarity_func: The similarity function that will be used to compare the adapted cases to the query.
        max_similarity_decrease: Maximum decrease in similarity allowed for an adapted case.

    Returns:
        The adapted casebase.
    """

    adaptation_func: AnyAdaptFunc[K, V]
    similarity_func: AnySimFunc[V, S]
    max_similarity_decrease: float | None = None
    processes: int = 1

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            **super(build, self).metadata,
            "adaptation_func": get_metadata(self.adaptation_func),
            "similarity_func": get_metadata(self.similarity_func),
            "max_similarity_decrease": self.max_similarity_decrease,
        }

    @override
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[Casebase[K, tuple[V | None, S]]]:
        sim_func = SimPairWrapper(self.similarity_func)
        adapted_casebases: list[dict[K, tuple[V | None, S]]] = []

        adaptation_func_signature = inspect_signature(self.adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adaptation_func = cast(
                AdaptMapFunc[K, V] | AdaptReduceFunc[K, V],
                self.adaptation_func,
            )

            if self.processes != 1:
                pool_processes = None if self.processes <= 0 else self.processes

                with Pool(pool_processes) as pool:
                    adaptation_results = pool.starmap(adaptation_func, pairs)
            else:
                adaptation_results = [
                    adaptation_func(casebase, query) for casebase, query in pairs
                ]

            if all(isinstance(item, tuple) for item in adaptation_results):
                adaptation_results = cast(Sequence[tuple[K, V]], adaptation_results)
                adapted_casebases = [
                    {adapted_key: (adapted_case, sim_func(adapted_case, query))}
                    for (_, query), (adapted_key, adapted_case) in zip(
                        pairs, adaptation_results, strict=True
                    )
                ]
            else:
                adaptation_results = cast(Sequence[Casebase[K, V]], adaptation_results)
                adapted_casebases = [
                    {
                        key: (adapted_case, sim_func(adapted_case, query))
                        for key, adapted_case in adapted_casebase.items()
                    }
                    for (_, query), adapted_casebase in zip(
                        pairs, adaptation_results, strict=True
                    )
                ]

        else:
            adaptation_func = cast(AdaptPairFunc[V], self.adaptation_func)
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

            adapted_casebases = [{} for _ in range(len(pairs))]

            for (idx, key), adapted_case, adapted_sim in zip(
                pairs_index, adapted_cases, adapted_sims, strict=True
            ):
                adapted_casebases[idx][key] = (adapted_case, adapted_sim)

        if self.max_similarity_decrease is not None:
            for (casebase, query), adapted_casebase in zip(
                pairs, adapted_casebases, strict=True
            ):
                for key, (_, adapted_sim) in adapted_casebase.items():
                    retrieved_sim = sim_func(casebase[key], query)

                    if (
                        unpack_sim(adapted_sim)
                        < unpack_sim(retrieved_sim) - self.max_similarity_decrease
                    ):
                        adapted_casebase[key] = (None, adapted_sim)

        return [
            self.postprocess(adapted_casebase) for adapted_casebase in adapted_casebases
        ]


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
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

    for reuser in reusers:
        queries_results = reuser(
            [
                (current_casebases[query_key], query)
                for query_key, query in queries.items()
            ]
        )
        step_queries = {
            query_key: QueryResultStep.build(
                query_result,
                current_casebases[query_key],
            )
            for query_key, query_result in zip(queries, queries_results, strict=True)
        }

        step = ResultStep(step_queries, get_metadata(reuser))
        steps.append(step)
        current_casebases = {
            query_key: step_queries[query_key].casebase for query_key in queries
        }

    return Result(steps)


def apply_single[V, S: Float](
    case: V,
    query: V,
    reusers: ReuserFunc[str, V, S] | Sequence[ReuserFunc[str, V, S]],
) -> Result[str, str, V, S]:
    """Applies a single query to a single case using reuser functions.

    Args:
        case: The case that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries({"default": case}, {"default": query}, reusers)


def apply_query[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[K, V, S] | Sequence[ReuserFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, reusers)


apply = apply_query
