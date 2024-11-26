from collections.abc import Mapping, Sequence

from ..helpers import (
    get_metadata,
)
from ..typing import (
    Casebase,
    Float,
    ReuserFunc,
)
from ._model import QueryResultStep, Result, ResultStep


def apply_queries[Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    reusers: ReuserFunc[C, V, S] | Sequence[ReuserFunc[C, V, S]],
    similarities: Mapping[C, S] | None = None,
) -> Result[Q, C, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase for the query.
        queries: The queries that will be used to adapt the casebase.
        reusers: The reuser functions that will be applied to the casebase.

    Returns:
        Returns an object of type Result
    """

    if not isinstance(reusers, Sequence):
        reusers = [reusers]

    steps: list[ResultStep[Q, C, V, S]] = []
    current_casebases: Mapping[Q, Mapping[C, V]] = {
        query_key: casebase for query_key in queries
    }
    current_similarities: Mapping[Q, Mapping[C, S] | None] = {
        query_key: similarities for query_key in queries
    }

    for reuser in reusers:
        queries_results = reuser(
            [
                (current_casebases[query_key], query, current_similarities[query_key])
                for query_key, query in queries.items()
            ]
        )
        step_queries = {
            query_key: QueryResultStep(
                adapted_sims,
                adapted_casebase,
            )
            for query_key, (adapted_casebase, adapted_sims) in zip(
                queries, queries_results, strict=True
            )
        }

        step = ResultStep(step_queries, get_metadata(reuser))
        steps.append(step)
        current_casebases = {
            query_key: step_queries[query_key].casebase for query_key in queries
        }
        current_similarities = {
            query_key: step_queries[query_key].similarities for query_key in queries
        }

    return Result(steps)


def apply_single[V, S: Float](
    case: V,
    query: V,
    reusers: ReuserFunc[str, V, S] | Sequence[ReuserFunc[str, V, S]],
    similarity: S | None = None,
) -> Result[str, str, V, S]:
    """Applies a single query to a single case using reuser functions.

    Args:
        case: The case that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(
        {"default": case},
        {"default": query},
        reusers,
        {"default": similarity} if similarity is not None else None,
    )


def apply_query[K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    reusers: ReuserFunc[K, V, S] | Sequence[ReuserFunc[K, V, S]],
    similarities: Mapping[K, S] | None = None,
) -> Result[str, K, V, S]:
    """Applies a single query to a casebase using reuser functions.

    Args:
        casebase: The casebase that will be used for the query.
        query: The query that will be applied to the case.
        reusers: The reuser functions that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, reusers, similarities)
