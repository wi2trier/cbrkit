__all__ = [
    "apply_queries",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .retrieval import Result as RetrievalResult
from .retrieval import apply_queries as apply_retrievers
from .reuse import Result as ReuseResult
from .reuse import apply_queries as apply_reusers
from .typing import Float, RetrieverFunc, ReuserFunc


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    retrieval: RetrievalResult[Q, C, V, S]
    reuse: ReuseResult[Q, C, V, S]

    def as_dict(self) -> dict[str, Any]:
        return {
            "retrieval": self.retrieval.as_dict(),
            "reuse": self.reuse.as_dict(),
        }


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    retrieval_result = apply_retrievers(casebase, queries, retrievers)
    reuse_result = apply_reusers(
        retrieval_result.casebase, queries, reusers, retrieval_result.similarities
    )

    return Result(retrieval_result, reuse_result)
