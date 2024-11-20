from typing import Any

from ..retrieval import Result, ResultStep
from ..typing import Float, QueryCaseMatrix
from ._common import compute


def retrieval_step[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    step: ResultStep[Q, C, Any, S],
) -> dict[str, float]:
    return compute(
        qrels,
        {query: entry.similarities for query, entry in step.by_query.items()},
    )


def retrieval[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    result: Result[Q, C, Any, S],
) -> list[dict[str, float]]:
    return [retrieval_step(qrels, step) for step in result.steps]
