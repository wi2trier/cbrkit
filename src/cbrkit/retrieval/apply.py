from collections.abc import Mapping
from timeit import default_timer
from typing import cast

from ..helpers import (
    get_logger,
    get_metadata,
    produce_factory,
    produce_sequence,
)
from ..model import QueryResultStep, Result, ResultStep
from ..typing import Float, InternalFunc, MaybeFactories, RetrieverFunc

logger = get_logger(__name__)


def apply_batches[Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V]],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies batches containing a casebase and a query using retriever functions.

    Args:
        batches: A mapping of queries to casebases and queries.
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Retrieval result object
    """
    retriever_factories = produce_sequence(retrievers)
    assert len(retriever_factories) > 0
    steps: list[ResultStep[Q, C, V, S]] = []
    current_batches: Mapping[Q, tuple[Mapping[C, V], V]] = batches

    loop_start_time = default_timer()

    for i, retriever_factory in enumerate(retriever_factories, start=1):
        retriever_func = produce_factory(retriever_factory)
        logger.info(f"Processing retriever {i}/{len(retriever_factories)}")

        resolved_values = list(current_batches.values())

        start_time = default_timer()
        queries_results = retriever_func(resolved_values)
        end_time = default_timer()

        step_queries: dict[Q, QueryResultStep[C, V, S]] = {}

        for (query_key, _), (_, query), (result_casebase, similarities) in zip(
            current_batches.items(), resolved_values, queries_results, strict=True
        ):
            filtered_casebase = {k: result_casebase[k] for k in similarities}
            step_queries[query_key] = QueryResultStep(
                similarities=similarities,
                casebase=filtered_casebase,
                query=query,
                duration=0.0,
            )

        current_batches = {
            query_key: (step_queries[query_key].casebase, step_queries[query_key].query)
            for query_key in current_batches
        }

        if not isinstance(retriever_func, InternalFunc):
            steps.append(
                ResultStep(
                    queries=step_queries,
                    metadata=get_metadata(retriever_func),
                    duration=end_time - start_time,
                )
            )

    logger.info("Finished processing all retrievers")

    return Result(steps=steps, duration=default_timer() - loop_start_time)


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies a single query to a Casebase using retriever functions.

    Args:
        casebase: The casebase that will be used to retrieve similar cases.
        queries: The queries that will be used to retrieve similar cases.
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Retrieval result object

    Examples:
        >>> import cbrkit
        >>> import polars as pl
        >>> df = pl.read_csv("./data/cars-1k.csv")
        >>> casebase = cbrkit.loaders.polars(df)
        >>> retriever = cbrkit.retrieval.build(
        ...     cbrkit.sim.attribute_value(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.linear(max=100000),
        ...             "year": cbrkit.sim.numbers.linear(max=50),
        ...             "miles": cbrkit.sim.numbers.linear(max=1000000),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     )
        ... )
        >>> result = cbrkit.retrieval.apply_queries(casebase, {"default": casebase[42]}, retriever)
    """
    return apply_batches(
        {query_key: (casebase, query) for query_key, query in queries.items()},
        retrievers,
    )


def apply_query[K, V, S: Float](
    casebase: Mapping[K, V],
    query: V,
    retrievers: MaybeFactories[RetrieverFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies a single query to a Casebase using retriever functions.

    Args:
        casebase: The casebase that will be used to retrieve similar cases.
        query: The query that will be used to retrieve similar cases.
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Retrieval result object
    """
    return apply_queries(
        casebase,
        {"default": query},
        retrievers,
    )


def apply_queries_indexed[Q, C, V, S: Float](
    queries: Mapping[Q, V],
    retrievers: MaybeFactories[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies queries using pre-indexed retrievers.

    This helper explicitly opts into indexed retrieval by passing an empty casebase.
    Every retriever must support indexed retrieval mode.
    """
    empty_casebase = cast(Mapping[C, V], {})

    return apply_batches(
        {query_key: (empty_casebase, query) for query_key, query in queries.items()},
        retrievers,
    )


def apply_query_indexed[K, V, S: Float](
    query: V,
    retrievers: MaybeFactories[RetrieverFunc[K, V, S]],
) -> Result[str, K, V, S]:
    """Applies a single query using pre-indexed retrievers."""
    return apply_queries_indexed(
        {"default": query},
        retrievers,
    )
