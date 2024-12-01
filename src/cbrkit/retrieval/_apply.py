from collections.abc import Mapping, Sequence

from ..helpers import (
    get_metadata,
)
from ..model import QueryResultStep, Result, ResultStep
from ..typing import (
    Float,
    RetrieverFunc,
)


def apply_pairs[Q, C, V, S: Float](
    pairs: Mapping[Q, tuple[Mapping[C, V], V]],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    steps: list[ResultStep[Q, C, V, S]] = []
    current_pairs: Mapping[Q, tuple[Mapping[C, V], V]] = pairs

    for retriever_func in retrievers:
        queries_results = retriever_func([pair for pair in current_pairs.values()])

        step_queries = {
            query_key: QueryResultStep.build(similarities, casebase, query)
            for (query_key, (casebase, query)), similarities in zip(
                current_pairs.items(), queries_results, strict=True
            )
        }

        steps.append(ResultStep(step_queries, get_metadata(retriever_func)))
        current_pairs = {
            query_key: (step_queries[query_key].casebase, step_queries[query_key].query)
            for query_key in current_pairs
        }

    return Result(steps)


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    retrievers: RetrieverFunc[C, V, S] | Sequence[RetrieverFunc[C, V, S]],
) -> Result[Q, C, V, S]:
    """Applies a single query to a Casebase using retriever functions.

    Args:
        casebase: The casebase that will be used to retrieve similar cases.
        queries: The queries that will be used to retrieve similar cases.
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Returns an object of type Result.

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
        ...             "manufacturer": cbrkit.sim.strings.taxonomy.load(
        ...                 "./data/cars-taxonomy.yaml",
        ...                 measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
        ...             ),
        ...             "miles": cbrkit.sim.numbers.linear(max=1000000),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     )
        ... )
        >>> result = cbrkit.retrieval.apply_queries(casebase, {"default": casebase[42]}, retriever)
    """
    return apply_pairs(
        {query_key: (casebase, query) for query_key, query in queries.items()},
        retrievers,
    )


def apply_query[K, V, S: Float](
    casebase: Mapping[K, V],
    query: V,
    retrievers: RetrieverFunc[K, V, S] | Sequence[RetrieverFunc[K, V, S]],
) -> Result[str, K, V, S]:
    return apply_queries(
        casebase,
        {"default": query},
        retrievers,
    )
