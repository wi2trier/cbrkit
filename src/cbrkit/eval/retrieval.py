from collections.abc import Sequence
from typing import Any, Literal

from ..helpers import normalize_and_scale, round, unpack_float
from ..retrieval import Result, ResultStep
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix
from .common import DEFAULT_METRICS, compute


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
    if max_qrel is None:
        return {
            query: {
                case: rank
                for rank, case in enumerate(reversed(entry.ranking), start=min_qrel)
            }
            for query, entry in result.queries.items()
        }

    sims = {
        query: {case: unpack_float(value) for case, value in entry.similarities.items()}
        for query, entry in result.queries.items()
    }
    if auto_scale:
        min_sim = min(min(entries.values()) for entries in sims.values())
        max_sim = max(max(entries.values()) for entries in sims.values())
    else:
        min_sim = 0.0
        max_sim = 1.0

    return {
        query: {
            case: round(
                normalize_and_scale(sim, min_sim, max_sim, min_qrel, max_qrel),
                round_mode,
            )
            for case, sim in entry.items()
        }
        for query, entry in sims.items()
    }


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
