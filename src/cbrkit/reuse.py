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
    SupportsParallelQueries,
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
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonDict

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        if len(self.queries) != 1:
            raise ValueError("The step contains multiple queries.")

        return next(iter(self.queries.values()))


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
    def default_query(self) -> QueryResultStep[C, V, S]:
        return self.final_step.default_query

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.default_query.similarities

    @property
    def casebase(self) -> Mapping[C, V]:
        return self.final_step.default_query.casebase

    @property
    def similarity(self) -> S:
        return self.final_step.default_query.similarity

    @property
    def case(self) -> V:
        return self.final_step.default_query.case

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

    def process(
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

        return {key: casebase[key] for key in ranking[: self.limit]}


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
    case_processes: int = 1
    query_processes: int = 1

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
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, tuple[V | None, S]]:
        sim_func = SimPairWrapper(self.similarity_func)
        adapted_casebase: dict[K, tuple[V | None, S]] = {}

        adaptation_func_signature = inspect_signature(self.adaptation_func)

        if "casebase" in adaptation_func_signature.parameters:
            adaptation_func = cast(
                AdaptMapFunc[K, V] | AdaptReduceFunc[K, V],
                self.adaptation_func,
            )
            adaptation_result = adaptation_func(casebase, query)

            if isinstance(adaptation_result, tuple):
                adapted_key, adapted_case = adaptation_result
                adapted_casebase = {
                    adapted_key: (adapted_case, sim_func(adapted_case, query))
                }
            else:
                adapted_casebase = {
                    key: (adapted_case, sim_func(adapted_case, query))
                    for key, adapted_case in adaptation_result.items()
                }

        else:
            adaptation_func = cast(AdaptPairFunc[V], self.adaptation_func)

            if self.case_processes != 1:
                pool_processes = (
                    None if self.case_processes <= 0 else self.case_processes
                )
                keys = list(casebase.keys())

                with Pool(pool_processes) as pool:
                    adapted_cases = pool.starmap(
                        adaptation_func,
                        ((casebase[key], query) for key in keys),
                    )
                    adapted_sims = pool.starmap(
                        sim_func,
                        ((case, query) for case in adapted_cases),
                    )

                adapted_casebase = {
                    key: (case, sim)
                    for key, case, sim in zip(
                        keys, adapted_cases, adapted_sims, strict=True
                    )
                }
            else:
                for key, case in casebase.items():
                    adapted_case = adaptation_func(case, query)
                    adapted_sim = sim_func(adapted_case, query)
                    adapted_casebase[key] = adapted_case, adapted_sim

        if self.max_similarity_decrease is not None:
            for key, (_, adapted_sim) in adapted_casebase.items():
                retrieved_sim = sim_func(casebase[key], query)

                if (
                    unpack_sim(adapted_sim)
                    < unpack_sim(retrieved_sim) - self.max_similarity_decrease
                ):
                    adapted_casebase[key] = (None, adapted_sim)

        return self.process(adapted_casebase)


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
        if isinstance(reuser, SupportsParallelQueries) and reuser.query_processes > 1:
            pool_processes = (
                None if reuser.query_processes <= 0 else reuser.query_processes
            )

            with Pool(pool_processes) as pool:
                reuse_results = pool.starmap(
                    reuser,
                    (
                        (current_casebases[query_key], query)
                        for query_key, query in queries.items()
                    ),
                )
                step_queries = {
                    query_key: QueryResultStep.build(
                        reuse_result, current_casebases[query_key]
                    )
                    for query_key, reuse_result in zip(
                        queries, reuse_results, strict=True
                    )
                }
        else:
            step_queries = {
                query_key: QueryResultStep.build(
                    reuser(current_casebases[query_key], query),
                    current_casebases[query_key],
                )
                for query_key, query in queries.items()
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
