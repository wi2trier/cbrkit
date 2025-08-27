from collections.abc import Sequence
from typing import Any, Literal

from ..helpers import unpack_float
from ..retrieval import Result, ResultStep
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix
from .common import DEFAULT_METRICS, compute, similarities_to_qrels


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


def retrieval_step_to_qrels[Q, C, S: Float](
    result: ResultStep[Q, C, Any, S],
    max_qrel: int | None = None,
    min_qrel: int = 1,
    round_mode: Literal["floor", "ceil", "nearest"] = "nearest",
    auto_scale: bool = True,
) -> QueryCaseMatrix[Q, C, int]:
    unpacked_sims = {
        query: {case: unpack_float(value) for case, value in entry.similarities.items()}
        for query, entry in result.queries.items()
    }
    return similarities_to_qrels(
        unpacked_sims,
        max_qrel,
        min_qrel,
        round_mode,
        auto_scale,
    )


def retrieval_to_qrels[Q, C, S: Float](
    result: Result[Q, C, Any, S],
    max_qrel: int = 5,
    min_qrel: int = 1,
    round_mode: Literal["floor", "ceil", "nearest"] = "nearest",
    auto_scale: bool = True,
) -> list[QueryCaseMatrix[Q, C, int]]:
    return [
        retrieval_step_to_qrels(
            step,
            max_qrel,
            min_qrel,
            round_mode,
            auto_scale,
        )
        for step in result.steps
    ]
