from collections.abc import Mapping
from timeit import default_timer

from ..helpers import get_logger, get_metadata, produce_factory, produce_sequence
from ..model import QueryResultStep, Result, ResultStep
from ..typing import (
    Casebase,
    Float,
    MaybeFactories,
    RetainerFunc,
)

logger = get_logger(__name__)


def apply_result[Q, C, V, S: Float](
    result: Result[Q, C, V, S] | ResultStep[Q, C, V, S],
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies retainer functions to a previous result.

    Args:
        result: The result whose cases may be retained.
        retainers: The retainer functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    if isinstance(result, ResultStep):
        result = Result(steps=[result], duration=0.0)

    if not produce_sequence(retainers):
        return result

    return apply_batches(
        {
            query_key: (entry.casebase, entry.query)
            for query_key, entry in result.queries.items()
        },
        retainers,
    )


def apply_batches[Q, C, V, S: Float](
    batch: Mapping[Q, tuple[Mapping[C, V], V]],
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies retainer functions to batches.

    Args:
        batch: A mapping of queries to (casebase, query) tuples.
        retainers: Retainer functions that will decide retention and storage.

    Returns:
        Returns an object of type Result.
    """
    retainer_factories = produce_sequence(retainers)
    steps: list[ResultStep[Q, C, V, S]] = []
    current_batches: Mapping[Q, tuple[Mapping[C, V], V]] = batch

    loop_start_time = default_timer()

    for i, retainer_factory in enumerate(retainer_factories, start=1):
        retainer_func = produce_factory(retainer_factory)
        logger.info(f"Processing retainer {i}/{len(retainer_factories)}")
        start_time = default_timer()
        queries_results = retainer_func(list(current_batches.values()))
        end_time = default_timer()

        step_queries = {
            query_key: QueryResultStep(
                similarities=retained_sims,
                casebase=retained_casebase,
                query=current_batches[query_key][1],
                duration=0.0,
            )
            for query_key, (retained_casebase, retained_sims) in zip(
                current_batches.keys(), queries_results, strict=True
            )
        }

        steps.append(
            ResultStep(
                queries=step_queries,
                metadata=get_metadata(retainer_func),
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
    retainers: MaybeFactories[RetainerFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies retainer functions to multiple queries.

    Args:
        casebase: The casebase containing cases.
        queries: The queries.
        retainers: The retainer functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_batches(
        {query_key: (casebase, query) for query_key, query in queries.items()},
        retainers,
    )


def apply_pair[V, S: Float](
    case: V,
    query: V,
    retainers: MaybeFactories[RetainerFunc[str, V, S]],
) -> Result[str, str, V, S]:
    """Applies retainer functions to a single case-query pair.

    Args:
        case: The case that may be retained.
        query: The query used for retention.
        retainers: The retainer functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(
        {"default": case},
        {"default": query},
        retainers,
    )


def apply_query[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    retainers: MaybeFactories[RetainerFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies retainer functions to a single query.

    Args:
        casebase: The casebase containing cases.
        query: The query.
        retainers: The retainer functions that will be applied.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, retainers)
