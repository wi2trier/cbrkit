from collections.abc import Sequence
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from typing import Any

from cbrkit.helpers import SimPairWrapper, get_metadata, unpack_sim
from cbrkit.typing import (
    AdaptPairFunc,
    AnySimFunc,
    Casebase,
    Float,
    JsonDict,
    ReuserFunc,
    SimPairFunc,
)

__all__ = [
    "build",
    "apply",
    "apply_map",
    "apply_reduce",
    "Result",
    "ResultStep",
]


@dataclass(slots=True, frozen=True)
class ResultStep[V, S: Float]:
    case: V | None
    similarity: S
    metadata: JsonDict

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["case"]

        return x


@dataclass(slots=True, frozen=True)
class Result[V, S: Float]:
    steps: list[ResultStep[V, S]]

    @property
    def final(self) -> ResultStep[V, S]:
        return self.steps[-1]

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


class build[V, S: Float](ReuserFunc[V, S]):
    adaptation_func: AdaptPairFunc[V]
    similarity_func: SimPairFunc[V, S]
    max_similarity_decrease: float | None

    def __init__(
        self,
        adaptation_func: AdaptPairFunc[V],
        similarity_func: AnySimFunc[V, S],
        max_similarity_decrease: float | None = None,
    ) -> None:
        self.adaptation_func = adaptation_func
        self.max_similarity_decrease = max_similarity_decrease
        self.similarity_func = SimPairWrapper(similarity_func)

    def __call__(self, case: V, query: V) -> tuple[V | None, S]:
        adapted_case = self.adaptation_func(case, query)
        new_similarity = self.similarity_func(adapted_case, query)

        if (
            self.max_similarity_decrease is not None
            and self.similarity_func is not None
        ):
            old_similarity = self.similarity_func(case, query)

            if (
                unpack_sim(new_similarity)
                < unpack_sim(old_similarity) - self.max_similarity_decrease
            ):
                return None, new_similarity

        return adapted_case, new_similarity


def apply[V, S: Float](
    case: V,
    query: V,
    reusers: ReuserFunc[V, S] | Sequence[ReuserFunc[V, S]],
) -> Result[V, S]:
    if not isinstance(reusers, Sequence):
        reusers = [reusers]

    steps: list[ResultStep[V, S]] = []
    current_case: V = case

    for reuser in reusers:
        adapted_case, adapted_sim = reuser(current_case, query)
        steps.append(
            ResultStep(
                case=adapted_case,
                similarity=adapted_sim,
                metadata=get_metadata(reuser),
            )
        )

        if adapted_case is not None:
            current_case = adapted_case

    return Result(steps)


def apply_map[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[V, S] | Sequence[ReuserFunc[V, S]],
    processes: int = 1,
) -> dict[K, Result[V, S]]:
    if processes != 1:
        pool_processes = None if processes <= 0 else processes
        keys = list(casebase.keys())

        with Pool(pool_processes) as pool:
            results = pool.starmap(
                apply,
                ((casebase[key], query, reusers) for key in keys),
            )

        return dict(zip(keys, results, strict=True))

    return {key: apply(case, query, reusers) for key, case in casebase.items()}


def apply_reduce[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[V, S] | Sequence[ReuserFunc[V, S]],
    processes: int = 1,
) -> Result[V, S]:
    """Return the best adapted case from the casebase."""

    results = apply_map(casebase, query, reusers, processes)
    return max(results.values(), key=lambda x: unpack_sim(x.similarity))
