from collections.abc import Sequence
from dataclasses import asdict, dataclass
from inspect import signature as inspect_signature
from multiprocessing import Pool
from typing import Any, cast

from cbrkit.helpers import SimPairWrapper, get_metadata, unpack_sim
from cbrkit.typing import (
    AdaptCompositionalFunc,
    AdaptPairFunc,
    AnyAdaptFunc,
    AnySimFunc,
    Casebase,
    Float,
    JsonDict,
    ReuserFunc,
)

__all__ = [
    "build",
    "apply",
    "Result",
    "ResultStep",
]


@dataclass(slots=True, frozen=True)
class ResultStep[K, V, S: Float]:
    key: K
    case: V | None
    similarity: S
    metadata: JsonDict

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["case"]

        return x


@dataclass(slots=True, frozen=True)
class Result[K, V, S: Float]:
    steps: list[ResultStep[K, V, S]]

    @property
    def final(self) -> ResultStep[K, V, S]:
        return self.steps[-1]

    @property
    def key(self) -> K:
        return self.final.key

    @property
    def similarity(self) -> S:
        return self.final.similarity

    @property
    def case(self) -> V | None:
        return self.final.case

    @property
    def metadata(self) -> JsonDict:
        return self.final.metadata

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for entry in x["steps"]:
            del entry["case"]

        return x


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](ReuserFunc[K, V, S]):
    adaptation_func: AnyAdaptFunc[K, V]
    similarity_func: AnySimFunc[V, S]
    max_similarity_decrease: float | None = None

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        processes: int,
    ) -> tuple[K, V | None, S]:
        sim_func = SimPairWrapper(self.similarity_func)

        adapt_signature = inspect_signature(self.adaptation_func)

        adapted_key: K
        adapted_case: V | None
        adapted_sim: S

        if "casebase" in adapt_signature.parameters:
            adapt_func = cast(AdaptCompositionalFunc[K, V], self.adaptation_func)
            adapted_key, adapted_case = adapt_func(casebase, query)
            adapted_sim = sim_func(adapted_case, query)

        else:
            adapt_func = cast(AdaptPairFunc[V], self.adaptation_func)
            adapted_casebase: dict[K, tuple[V | None, S]] = {}

            if processes != 1:
                pool_processes = None if processes <= 0 else processes
                keys = list(casebase.keys())

                with Pool(pool_processes) as pool:
                    adapted_cases = pool.starmap(
                        adapt_func,
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
                    adapted_case = adapt_func(case, query)
                    adapted_sim = sim_func(adapted_case, query)
                    adapted_casebase[key] = adapted_case, adapted_sim

            adapted_key, (adapted_case, adapted_sim) = max(
                adapted_casebase.items(),
                key=lambda x: unpack_sim(x[1][1]),
            )

        if self.max_similarity_decrease is not None:
            old_sim = sim_func(casebase[adapted_key], query)

            if (
                unpack_sim(adapted_sim)
                < unpack_sim(old_sim) - self.max_similarity_decrease
            ):
                adapted_case = None

        return adapted_key, adapted_case, adapted_sim


def apply[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[K, V, S] | Sequence[ReuserFunc[K, V, S]],
    processes: int = 1,
) -> Result[K, V, S]:
    if not isinstance(reusers, Sequence):
        reusers = [reusers]

    steps: list[ResultStep[K, V, S]] = []
    current_casebase: Casebase[K, V] = casebase

    for reuser in reusers:
        # TODO: move logic from "build" to here so that the entire case base is adapted and can be forwarded to the next step
        adapted_key, adapted_case, addapted_sim = reuser(
            current_casebase, query, processes
        )
        steps.append(
            ResultStep(adapted_key, adapted_case, addapted_sim, get_metadata(reuser))
        )

        if adapted_case is not None:
            current_casebase = {adapted_key: adapted_case}
        else:
            current_casebase = {adapted_key: casebase[adapted_key]}

    return Result(steps)
