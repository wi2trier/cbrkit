from collections.abc import Sequence
from typing import Any

from ..retrieval import Result, ResultStep
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix
from ._common import DEFAULT_METRICS, compute


def retrieval_step[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    step: ResultStep[Q, C, Any, S],
    metrics: Sequence[str] = DEFAULT_METRICS,
    metric_funcs: dict[str, EvalMetricFunc] | None = None,
) -> dict[str, float]:
    return compute(
        qrels,
        {query: entry.similarities for query, entry in step.queries.items()},
        metrics,
        metric_funcs,
    )


def retrieval[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    result: Result[Q, C, Any, S],
    metrics: Sequence[str] = DEFAULT_METRICS,
    metric_funcs: dict[str, EvalMetricFunc] | None = None,
) -> list[dict[str, float]]:
    return [
        retrieval_step(
            qrels,
            step,
            metrics,
            metric_funcs,
        )
        for step in result.steps
    ]
