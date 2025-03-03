from collections.abc import Mapping
from timeit import default_timer

from .model import CycleResult as Result
from .retrieval import apply_batches as apply_retieval_batches
from .retrieval import apply_queries as apply_retrieval_queries
from .reuse import apply_result as apply_reuse_result
from .typing import Float, MaybeFactories, RetrieverFunc, ReuserFunc

__all__ = [
    "apply_queries",
    "apply_batches",
    "Result",
]


def apply_batches[Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V]],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
    reusers: MaybeFactories[ReuserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    start_time = default_timer()
    retrieval_result = apply_retieval_batches(batches, retrievers)
    reuse_result = apply_reuse_result(retrieval_result, reusers)
    end_time = default_timer()

    return Result(
        retrieval=retrieval_result, reuse=reuse_result, duration=end_time - start_time
    )


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
    reusers: MaybeFactories[ReuserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    start_time = default_timer()
    retrieval_result = apply_retrieval_queries(casebase, queries, retrievers)
    reuse_result = apply_reuse_result(retrieval_result, reusers)
    end_time = default_timer()

    return Result(
        retrieval=retrieval_result, reuse=reuse_result, duration=end_time - start_time
    )
