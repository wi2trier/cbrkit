from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .model import Result as BaseResult
from .model import ResultStep as BaseResultStep
from .retrieval import apply_batches as apply_retieval_batches
from .retrieval import apply_queries as apply_retrieval_queries
from .reuse import apply_result as apply_reuse_result
from .typing import Float, RetrieverFunc, ReuserFunc

__all__ = [
    "apply_queries",
    "apply_batches",
    "Result",
]


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    retrieval: BaseResult[Q, C, V, S]
    reuse: BaseResult[Q, C, V, S]

    @property
    def final_step(self) -> BaseResultStep[Q, C, V, S]:
        if len(self.reuse.steps) > 0:
            return self.reuse.final_step

        return self.retrieval.final_step

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "reuse": self.reuse.as_dict(),
        }


def apply_batches[Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V]],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    retrieval_result = apply_retieval_batches(batches, retrievers)
    reuse_result = apply_reuse_result(retrieval_result, reusers)

    return Result(retrieval_result, reuse_result)


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    retrieval_result = apply_retrieval_queries(casebase, queries, retrievers)
    reuse_result = apply_reuse_result(retrieval_result, reusers)

    return Result(retrieval_result, reuse_result)
