from collections.abc import Mapping
from timeit import default_timer

from ..helpers import get_logger, get_metadata, produce_factory, produce_sequence
from ..model import QueryResultStep, Result, ResultStep
from ..typing import (
    Casebase,
    Float,
    MaybeFactories,
    ReviserFunc,
)

logger = get_logger(__name__)


def apply_result[Q, C, V, S: Float](
    result: Result[Q, C, V, S] | ResultStep[Q, C, V, S],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies reviser functions to a previous result.

    Args:
        result: The result whose solutions will be revised.
        revisers: The reviser functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    if isinstance(result, ResultStep):
        result = Result(steps=[result], duration=0.0)

    if not produce_sequence(revisers):
        return result

    return apply_batches(
        {
            query_key: (entry.casebase, entry.query)
            for query_key, entry in result.queries.items()
        },
        revisers,
    )


def apply_batches[Q, C, V, S: Float](
    batch: Mapping[Q, tuple[Mapping[C, V], V]],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies batches containing a casebase and a query using reviser functions.

    Args:
        batch: A mapping of queries to casebases and queries.
        revisers: Reviser functions that will assess and optionally repair solutions.

    Returns:
        Returns an object of type Result.
    """
    reviser_factories = produce_sequence(revisers)
    steps: list[ResultStep[Q, C, V, S]] = []
    current_batches: Mapping[Q, tuple[Mapping[C, V], V]] = batch

    loop_start_time = default_timer()

    for i, reviser_factory in enumerate(reviser_factories, start=1):
        reviser_func = produce_factory(reviser_factory)
        logger.info(f"Processing reviser {i}/{len(reviser_factories)}")
        start_time = default_timer()
        queries_results = reviser_func(list(current_batches.values()))
        end_time = default_timer()

        step_queries = {
            query_key: QueryResultStep(
                similarities=revised_sims,
                casebase=revised_casebase,
                query=current_batches[query_key][1],
                duration=0.0,
            )
            for query_key, (revised_casebase, revised_sims) in zip(
                current_batches.keys(), queries_results, strict=True
            )
        }

        steps.append(
            ResultStep(
                queries=step_queries,
                metadata=get_metadata(reviser_func),
                duration=end_time - start_time,
            )
        )
        current_batches = {
            query_key: (step_queries[query_key].casebase, step_queries[query_key].query)
            for query_key in current_batches.keys()
        }

    return Result(steps=steps, duration=default_timer() - loop_start_time)


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    revisers: MaybeFactories[ReviserFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies queries to a casebase using reviser functions.

    Args:
        casebase: The casebase containing solutions to revise.
        queries: The queries used for revision.
        revisers: The reviser functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_batches(
        {query_key: (casebase, query) for query_key, query in queries.items()},
        revisers,
    )


def apply_pair[V, S: Float](
    case: V,
    query: V,
    revisers: MaybeFactories[ReviserFunc[str, V, S]],
) -> Result[str, str, V, S]:
    """Applies a single query to a single case using reviser functions.

    Args:
        case: The case that will be revised.
        query: The query used for revision.
        revisers: The reviser functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(
        {"default": case},
        {"default": query},
        revisers,
    )


def apply_query[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    revisers: MaybeFactories[ReviserFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies a single query to a casebase using reviser functions.

    Args:
        casebase: The casebase containing solutions to revise.
        query: The query used for revision.
        revisers: The reviser functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, revisers)
