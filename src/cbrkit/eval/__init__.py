"""Evaluation metrics for assessing retrieval quality based on ``ranx``.

This module provides functions for computing Information Retrieval metrics
on CBRkit retrieval results.
Standard metrics from ``ranx`` (e.g., ``precision``, ``recall``, ``f1``) are supported
alongside two CBRkit-specific metrics.

Functions:
    ``compute``: Computes metrics given ground truth (qrels), retrieval scores (run),
    and a list of metric names.
    Supports cutoff notation (e.g., ``precision@5``).
    ``retrieval``: Evaluates a full retrieval ``Result`` directly.
    ``retrieval_step``: Evaluates a single retrieval ``ResultStep``.
    ``generate_metrics``: Generates metric name strings for multiple cutoff points.
    ``parse_metric``: Parses a metric string like ``"precision@5"`` into its components.
    ``compute_score_metrics``: Computes score-based (non-ranking) metrics.
    ``similarities_to_qrels``: Converts similarity maps to relevance judgments (qrels).
    ``retrieval_to_qrels`` / ``retrieval_step_to_qrels``: Converts retrieval results to qrels.

CBRkit-specific Metrics:
    ``correctness``: Measures how well the ranking preserves relevance ordering (-1 to 1).
    ``completeness``: Measures what fraction of relevance pairs are preserved (0 to 1).

Custom metrics can be provided via the ``metric_funcs`` parameter of ``compute``.
See ``cbrkit.typing.EvalMetricFunc`` for the expected signature.

For the full list of ``ranx`` metrics, see: https://amenra.github.io/ranx/metrics/

Example:
    >>> results = compute(
    ...     qrels={"q1": {"c1": 2, "c2": 1}},
    ...     run={"q1": {"c1": 0.9, "c2": 0.5}},
    ...     metrics=["precision@5", "recall@10"],
    ... )
"""

from .common import (
    compute,
    compute_score_metrics,
    generate_metrics,
    parse_metric,
    similarities_to_qrels,
)
from .retrieval import (
    retrieval,
    retrieval_step,
    retrieval_step_to_qrels,
    retrieval_to_qrels,
)

__all__ = [
    "compute",
    "generate_metrics",
    "parse_metric",
    "compute_score_metrics",
    "similarities_to_qrels",
    "retrieval",
    "retrieval_step",
    "retrieval_step_to_qrels",
    "retrieval_to_qrels",
]
