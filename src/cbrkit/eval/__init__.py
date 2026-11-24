"""Evaluation metrics for assessing retrieval quality.

Metrics are typed objects that score the ranking of one query:
`Metric` names a built-in metric with its cutoff `k` (all cases if None) and minimal relevance,
such as `Metric(name="precision", k=5)`, and any hashable function of the `cbrkit.typing.EvalMetricFunc` protocol
is a custom metric.
`compute` aggregates the values of every metric over the queries, by their mean unless an `aggregation` is given.
The standard metrics are `hits`, `hit_rate`, `precision`, `recall`, `f1`, `r_precision`,
`mrr`, `map`, `dcg`, `dcg_burges`, `ndcg`, and `ndcg_burges`, see `MetricName`.

Functions:
- `compute`: Computes metrics given ground truth (qrels) and retrieval scores (run).
- `evaluate`: Evaluates the results of a cycle, a `Result` (its last step) or one of its steps.
- `generate_metrics`: Generates metrics for multiple cutoff points and relevance levels.
- `compute_score_metrics`: Evaluates scores against expected scores with metrics such as correlations.
- `similarities_to_qrels`: Converts similarity maps to relevance judgments (qrels).
- `to_qrels`: Converts the results of a cycle to qrels.

CBRkit-specific Metrics:
- `correctness`: Measures how well the ranking preserves relevance ordering (-1 to 1).
- `completeness`: Measures what fraction of relevance pairs are preserved (0 to 1).
- `kendall_tau`: Kendall's tau between relevance and retrieval scores (-1 to 1).
- `mean_score`: Mean retrieval score of the top-k results.

Example:
    >>> results = compute(
    ...     qrels={"q1": {"c1": 2, "c2": 1}},
    ...     run={"q1": {"c1": 0.9, "c2": 0.5}},
    ...     metrics=[Metric(name="precision", k=5), Metric(name="recall", k=10)],
    ... )
    >>> results[Metric(name="recall", k=10)]
    1.0
"""

from .common import (
    DEFAULT_METRICS,
    Metric,
    MetricName,
    ScoreMetricFunc,
    compute,
    compute_score_metrics,
    generate_metrics,
    similarities_to_qrels,
)
from .retrieval import evaluate, to_qrels

__all__ = [
    "DEFAULT_METRICS",
    "Metric",
    "MetricName",
    "ScoreMetricFunc",
    "compute",
    "compute_score_metrics",
    "evaluate",
    "generate_metrics",
    "similarities_to_qrels",
    "to_qrels",
]
