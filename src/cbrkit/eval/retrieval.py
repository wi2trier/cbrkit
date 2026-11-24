import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from ..model import QueryResult
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix
from .common import DEFAULT_METRICS, compute, similarities_to_qrels

__all__ = ["evaluate", "to_qrels"]


def evaluate[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    results: Mapping[Q, QueryResult[C, Any, S]],
    metrics: Sequence[EvalMetricFunc] = DEFAULT_METRICS,
    aggregation: Callable[[Sequence[float]], float] = statistics.fmean,
) -> dict[EvalMetricFunc, float]:
    """Evaluate the results of a cycle against relevance judgments, see `compute`.

    Args:
        qrels: Relevance of the cases for every query, higher is more relevant.
        results: The result of every query, such as a `cbrkit.model.Result` (its last step)
            or one of its `cbrkit.model.Step`.
        metrics: Metrics such as `Metric` or custom functions of the `cbrkit.typing.EvalMetricFunc` protocol.
        aggregation: Aggregates the values of a metric over the queries.
    """
    return compute(
        qrels,
        {query: result.similarities for query, result in results.items()},
        metrics,
        aggregation,
    )


def to_qrels[Q, C, S: Float](
    results: Mapping[Q, QueryResult[C, Any, S]],
    max_qrel: int | None = None,
    min_qrel: int = 1,
    round_mode: Literal["floor", "ceil", "nearest"] = "nearest",
    auto_scale: bool = True,
) -> QueryCaseMatrix[Q, C, int]:
    """Convert the similarities of the results of a cycle to integer relevance judgments, see `similarities_to_qrels`.

    Args:
        results: The result of every query, such as a `cbrkit.model.Result` (its last step)
            or one of its `cbrkit.model.Step`.
        max_qrel: Largest relevance, which the most similar case gets,
            the cases are ranked from `min_qrel` upwards by their order if None.
        min_qrel: Smallest relevance.
        round_mode: How scaled similarities are rounded to relevances.
        auto_scale: Scale by the smallest and largest similarity instead of between 0 and 1.
    """
    return similarities_to_qrels(
        {query: result.similarities for query, result in results.items()},
        max_qrel,
        min_qrel,
        round_mode,
        auto_scale,
    )
