from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import Any, Literal, cast, override

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
    "apply",
    "apply_single",
    "Result",
    "ResultStep",
]


@dataclass(slots=True, frozen=True)
class ResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    casebase: Casebase[K, V]
    metadata: JsonDict

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
class Result[K, V, S: Float]:
    steps: list[ResultStep[K, V, S]]

    @property
    def final(self) -> ResultStep[K, V, S]:
        return self.steps[-1]

    @property
    def similarities(self) -> SimMap[K, S]:
        return self.final.similarities

    @property
    def similarity(self) -> S:
        return self.final.similarity

    @property
    def casebase(self) -> Casebase[K, V]:
        return self.final.casebase

    @property
    def case(self) -> V:
        return self.final.case

    @property
    def metadata(self) -> JsonDict:
        return self.final.metadata

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for entry in x["steps"]:
            del entry["casebase"]

        return x


@dataclass(slots=True, frozen=True, kw_only=True)
class base_reuser[K, V, S: Float](ReuserFunc[K, V, S], SupportsMetadata):
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

        return {key: casebase[key] for key in ranking[: self.limit]}


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](base_reuser[K, V, S]):
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

    @property
    @override
    def metadata(self) -> JsonDict:
        return {
            **super(build, self).metadata,
            "adaptation_func": get_metadata(self.adaptation_func),
            "similarity_func": get_metadata(self.similarity_func),
            "max_similarity_decrease": self.max_similarity_decrease,
        }

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        processes: int,
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

            if processes != 1:
                pool_processes = None if processes <= 0 else processes
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

        return self.postprocess(adapted_casebase)


def apply_single[V, S: Float](
    case: V,
    query: V,
    reusers: ReuserFunc[str, V, S] | Sequence[ReuserFunc[str, V, S]],
    processes: int = 1,
) -> Result[str, V, S]:
    """Applies a single query to a single case using reuser functions.

    Args:
        case: The case that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.
        processes: Number of CPUs that will be used for multiprocessing.
            If 1, a regular loop will be used.
            If 0, the number of processes will be equal to the number of CPUs.
            Negative values will be treated as 0.

    Returns:
        Returns an object of type Result.
    """
    return apply({"default": case}, query, reusers, processes)


def apply[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[K, V, S] | Sequence[ReuserFunc[K, V, S]],
    processes: int = 1,
) -> Result[K, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase for the query.
        query: The query that will be applied to the casebase.
        reusers: The reuser functions that will be applied to the casebase.
        processes: Number of CPUs that will be used for multiprocessing.
            If 1, a regular loop will be used.
            If 0, the number of processes will be equal to the number of CPUs.
            Negative values will be treated as 0.

    Returns:
        Returns an object of type Result
    """

    if not isinstance(reusers, Sequence):
        reusers = [reusers]

    steps: list[ResultStep[K, V, S]] = []
    current_casebase = casebase

    for reuser in reusers:
        reuse_results = reuser(current_casebase, query, processes)
        adapted_casebase: dict[K, V] = {}
        adapted_similarities: dict[K, S] = {}

        for key, (adapted_case, adapted_sim) in reuse_results.items():
            adapted_casebase[key] = (
                adapted_case if adapted_case is not None else current_casebase[key]
            )
            adapted_similarities[key] = adapted_sim

        steps.append(
            ResultStep(adapted_similarities, adapted_casebase, get_metadata(reuser))
        )

        current_casebase = adapted_casebase

    return Result(steps)


def mapply[QK, CK, V, S: Float](
    casebase: Casebase[CK, V],
    queries: Mapping[QK, V],
    reusers: ReuserFunc[CK, V, S] | Sequence[ReuserFunc[CK, V, S]],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[QK, Result[CK, V, S]]:
    """Applies multiple queries to a Casebase using reuser functions.

    Args:
        casebase: The casebase for the query.
        queries: The queries that will be applied to the casebase
        reusers: The reuser functions that will be applied to the casebase.
        processes: Number of CPUs that will be used for multiprocessing.
            If 1, a regular loop will be used.
            If 0, the number of processes will be equal to the number of CPUs.
            Negative values will be treated as 0.
        parallel: Strategy for parallelization.
            If "queries", each query will be processed in parallel,
            if "casebase" the whole casebase will be processed in parallel.

    Returns:
        Returns an object of type Result.
    """

    if processes != 1 and parallel == "queries":
        pool_processes = None if processes <= 0 else processes
        keys = list(queries.keys())

        with Pool(pool_processes) as pool:
            results = pool.starmap(
                apply,
                ((casebase, queries[key], reusers) for key in keys),
            )

        return dict(zip(keys, results, strict=True))

    return {
        key: apply(casebase, value, reusers, processes)
        for key, value in queries.items()
    }
