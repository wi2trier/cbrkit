from collections.abc import Mapping
from timeit import default_timer

from .model import CycleResult as Result
from .model import Result as PhaseResult
from .retain import apply_result as apply_retain_result
from .retrieval import apply_batches as apply_retrieval_batches
from .retrieval import apply_queries as apply_retrieval_queries
from .reuse import apply_result as apply_reuse_result
from .revise import apply_result as apply_revise_result
from .typing import (
    Float,
    MaybeFactories,
    RetainerFunc,
    RetrieverFunc,
    ReuserFunc,
    ReviserFunc,
)

__all__ = [
    "apply_queries",
    "apply_batches",
    "apply_query",
    "Result",
]


def _complete_cycle[Q, C, V, S: Float](
    retrieval_result: PhaseResult[Q, C, V, S],
    reusers: MaybeFactories[ReuserFunc[C, V, S]],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
    start_time: float,
) -> Result[Q, C, V, S]:
    reuse_result = apply_reuse_result(retrieval_result, reusers)
    revise_result = apply_revise_result(reuse_result, revisers)
    retain_result = apply_retain_result(revise_result, retainers)

    return Result(
        retrieval=retrieval_result,
        reuse=reuse_result,
        revise=revise_result,
        retain=retain_result,
        duration=default_timer() - start_time,
    )


def apply_batches[Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V]],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
    reusers: MaybeFactories[ReuserFunc[C, V, S]],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies a full CBR cycle to batches using all phase functions.

    Args:
        batches: A mapping of queries to (casebase, query) tuples.
        retrievers: Retriever functions for the retrieve phase.
        reusers: Reuser functions for the reuse phase.
        revisers: Reviser functions for the revise phase.
        retainers: Retainer functions for the retain phase.

    Returns:
        A cycle result containing results from all applied phases.
    """
    start_time = default_timer()
    retrieval_result = apply_retrieval_batches(batches, retrievers)

    return _complete_cycle(retrieval_result, reusers, revisers, retainers, start_time)


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
    reusers: MaybeFactories[ReuserFunc[C, V, S]],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies a full CBR cycle to queries against a casebase.

    Args:
        casebase: The casebase to use for retrieval.
        queries: The queries to process through the cycle.
        retrievers: Retriever functions for the retrieve phase.
        reusers: Reuser functions for the reuse phase.
        revisers: Reviser functions for the revise phase.
        retainers: Retainer functions for the retain phase.

    Returns:
        A cycle result containing results from all applied phases.
    """
    start_time = default_timer()
    retrieval_result = apply_retrieval_queries(casebase, queries, retrievers)

    return _complete_cycle(retrieval_result, reusers, revisers, retainers, start_time)


def apply_query[K, V, S: Float](
    casebase: Mapping[K, V],
    query: V,
    retrievers: MaybeFactories[RetrieverFunc[K, V, S]],
    reusers: MaybeFactories[ReuserFunc[K, V, S]],
    revisers: MaybeFactories[ReviserFunc[K, V, S]],
    retainers: MaybeFactories[RetainerFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies a full CBR cycle to a single query against a casebase.

    Args:
        casebase: The casebase to use for retrieval.
        query: The query to process through the cycle.
        retrievers: Retriever functions for the retrieve phase.
        reusers: Reuser functions for the reuse phase.
        revisers: Reviser functions for the revise phase.
        retainers: Retainer functions for the retain phase.

    Returns:
        A cycle result containing results from all applied phases.
    """
    return apply_queries(
        casebase,
        {"default": query},
        retrievers,
        reusers,
        revisers,
        retainers,
    )
