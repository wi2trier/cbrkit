from collections.abc import Mapping

from .. import model
from ..helpers import get_logger, get_metadata
from ..typing import Casebase, Float, SynthesizerFunc
from .model import QueryResultStep, Result, ResultStep

logger = get_logger(__name__)


def apply_result[R, Q, C, V, S: Float](
    result: model.Result[Q, C, V, S] | model.ResultStep[Q, C, V, S],
    synthesizer: SynthesizerFunc[R, C, V, S],
) -> Result[Q, R]:
    return apply_batches(
        {
            key: (entry.casebase, entry.query, entry.similarities)
            for key, entry in result.queries.items()
        },
        synthesizer,
    )


def apply_batches[R, Q, C, V, S: Float](
    batches: Mapping[Q, tuple[Mapping[C, V], V | None, Mapping[C, S] | None]],
    synthesizer: SynthesizerFunc[R, C, V, S],
) -> Result[Q, R]:
    logger.info(f"Processing {len(batches)} batches")
    synthesis_results = synthesizer(list(batches.values()))

    return Result(
        [
            ResultStep(
                {
                    key: QueryResultStep(value)
                    for key, value in zip(
                        batches.keys(), synthesis_results, strict=True
                    )
                },
                get_metadata(synthesizer),
            )
        ]
    )


def apply_queries[R, Q, C, V, S: Float](
    casebase: Mapping[C, V],
    queries: Mapping[Q, V],
    synthesizer: SynthesizerFunc[R, C, V, S],
) -> Result[Q, R]:
    """Applies multiple queries to a casebase using synthesis functions.

    Args:
        casebase: The casebase for the query.
        queries: The queries that will be used to adapt the casebase.
        synthesizer: The synthesis function that will be applied to the casebase.

    Returns:
        Returns an object of type Result
    """
    return apply_batches(
        {query_key: (casebase, query, None) for query_key, query in queries.items()},
        synthesizer,
    )


def apply_query[R, K, V, S: Float](
    casebase: Casebase[K, V],
    query: V,
    synthesizer: SynthesizerFunc[R, K, V, S],
) -> Result[str, R]:
    """Applies a single query to a casebase using synthesis functions.

    Args:
        casebase: The casebase that will be used for the query.
        query: The query that will be applied to the case.
        synthesizer: The synthesis function that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_queries(casebase, {"default": query}, synthesizer)


def apply_casebase[R, K, V, S: Float](
    casebase: Casebase[K, V],
    synthesizer: SynthesizerFunc[R, K, V, S],
) -> Result[str, R]:
    """Applies a single query to a casebase using synthesis functions.

    Args:
        casebase: The casebase that will be used for the query.
        synthesizer: The synthesis function that will be applied to the case.

    Returns:
        Returns an object of type Result.
    """
    return apply_batches(
        {"default": (casebase, None, None)},
        synthesizer,
    )
