import itertools
import logging
import math
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

import numpy as np
from pydantic import Field

from .._util import sim_map2ranking
from ..schema import serializable
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix

__all__ = [
    "DEFAULT_METRICS",
    "QUERY_METRICS",
    "Metric",
    "MetricName",
    "QueryMetricFunc",
    "ScoreMetricFunc",
    "average_precision",
    "completeness",
    "compute",
    "compute_score_metrics",
    "concordances",
    "correctness",
    "f1",
    "generate_metrics",
    "hit_rate",
    "hits",
    "kendall_tau",
    "mean_score",
    "precision",
    "r_precision",
    "recall",
    "reciprocal_rank",
    "similarities_to_qrels",
]

logger = logging.getLogger(__name__)


type ScoreMetricFunc = Callable[[Sequence[float], Sequence[float]], float]
"""Compares the expected and the actual scores of the same cases, such as a correlation or an error."""


def compute_score_metrics[Q, C](
    expected: QueryCaseMatrix[Q, C, Float],
    actual: QueryCaseMatrix[Q, C, Float],
    metrics: Sequence[ScoreMetricFunc],
    aggregation: Callable[[Sequence[float]], float] = statistics.fmean,
) -> dict[ScoreMetricFunc, float]:
    """Evaluate scores against expected scores, the counterpart of `compute` for scores instead of rankings.

    Every metric receives the scores of the cases of a query that both matrices contain,
    and its values over the shared queries are aggregated, NaN if there is none.

    Args:
        expected: Expected score of the cases for every query.
        actual: Actual score of the cases for every query, such as their similarities.
        metrics: Hashable functions of the expected and the actual scores, which key the results.
        aggregation: Aggregates the values of a metric over the queries.

    Examples:
        >>> def error(expected, actual):
        ...     return max(abs(x - y) for x, y in zip(expected, actual))
        >>> compute_score_metrics({"q": {"a": 1.0, "b": 0.5}}, {"q": {"a": 0.75, "c": 0.0}}, [error])[error]
        0.25
    """
    queries = expected.keys() & actual.keys()
    values: dict[ScoreMetricFunc, float] = {}

    for metric in metrics:
        scores = []

        for query in queries:
            cases = sorted(expected[query].keys() & actual[query].keys(), key=str)
            scores.append(
                metric(
                    [float(expected[query][case]) for case in cases],
                    [float(actual[query][case]) for case in cases],
                )
            )

        values[metric] = aggregation(scores) if scores else float("nan")

    return values


def concordances(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> tuple[int, int, int]:
    """Compute the number of concordant and discordant pairs in a ranking.

    Args:
        qrel: The query relevance judgments.
        run: The ranking produced by the system.
        k: The number of top documents to consider, all if None.
        relevance_level: The relevance level to consider.

    Returns:
        A tuple with the number of concordant pairs, discordant pairs, and total pairs.

    Examples:
        >>> qrel = {
        ...     "case1": 3,
        ...     "case2": 4,
        ...     "case3": 8,
        ...     "case4": 5,
        ...     "case5": 10,
        ... }
        >>> run1 = {
        ...     "case1": 0.3,
        ...     "case2": 0.1,
        ...     "case3": 0.65,
        ...     "case4": 0.55,
        ...     "case5": 0.8,
        ... }
        >>> concordances(qrel, run1, None, 1)
        (9, 1, 10)
        >>> run2 = {
        ...     "case1": 0.6,
        ...     "case2": 0.4,
        ...     "case3": 0.55,
        ...     "case4": 0.7,
        ...     "case5": 0.58,
        ... }
        >>> concordances(qrel, run2, None, 1)
        (5, 5, 10)
    """
    relevant = _relevant(qrel, relevance_level)
    ranked = set(sim_map2ranking(run, k))
    # Pairs of distinct relevance are all pairs minus the tied ones, counted without a matrix over every judgment.
    pairs = (
        len(relevant) ** 2 - sum(n * n for n in Counter(relevant.values()).values())
    ) // 2
    retrieved = [doc for doc in relevant if doc in ranked]
    rels = np.array([relevant[doc] for doc in retrieved], dtype=np.float64)
    scores = np.array([run[doc] for doc in retrieved], dtype=np.float64)

    rel_order = np.sign(rels[:, None] - rels[None, :])
    covered = np.triu(rel_order != 0, k=1)
    concordant = int(
        np.sum(covered & (rel_order == np.sign(scores[:, None] - scores[None, :])))
    )

    return concordant, int(np.sum(covered)) - concordant, pairs


type QueryMetricFunc = Callable[
    [Mapping[str, int], Mapping[str, float], int | None, int], float | None
]


def _relevant(qrel: Mapping[str, int], relevance_level: int) -> dict[str, int]:
    return {doc: rel for doc, rel in qrel.items() if rel >= relevance_level}


def _dcg(gains: Sequence[float], burges: bool) -> float:
    values = np.asarray(gains, dtype=np.float64)

    if burges:
        values = np.exp2(values) - 1

    return float(np.sum(values / np.log2(np.arange(2, len(values) + 2))))


def hits(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Count the relevant documents among the top-k results.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    relevant = _relevant(qrel, relevance_level)

    return float(sum(doc in relevant for doc in sim_map2ranking(run, k)))


def hit_rate(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Return 1 if any relevant document is among the top-k results, else 0.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    return float(hits(qrel, run, k, relevance_level) > 0)


def precision(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Fraction of the top-k results that are relevant, k defaults to the run length.

    Follows Manning, Raghavan, and Schütze, Introduction to Information Retrieval, 2008, ch. 8.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> precision({"a": 1, "b": 1}, {"a": 0.9, "c": 0.5}, k=4, relevance_level=1)
        0.25
    """
    cutoff = len(run) if k is None else k

    return hits(qrel, run, k, relevance_level) / cutoff if cutoff else 0.0


def recall(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Fraction of the relevant documents found among the top-k results.

    Follows Manning, Raghavan, and Schütze, Introduction to Information Retrieval, 2008, ch. 8.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    relevant = _relevant(qrel, relevance_level)

    return hits(qrel, run, k, relevance_level) / len(relevant) if relevant else 0.0


def f1(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Harmonic mean of precision and recall (balanced F measure, Manning et al. 2008, ch. 8).

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    p = precision(qrel, run, k, relevance_level)
    r = recall(qrel, run, k, relevance_level)

    return 2 * p * r / (p + r) if p + r else 0.0


def r_precision(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Precision at the number of relevant documents, ignoring k (Manning et al. 2008, ch. 8).

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    relevant = _relevant(qrel, relevance_level)

    return precision(qrel, run, len(relevant), relevance_level) if relevant else 0.0


def reciprocal_rank(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Inverse rank of the first relevant document among the top-k results.

    Averaged over queries, this is the mean reciprocal rank (Voorhees, TREC-8 QA track, 1999).

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.
    """
    relevant = _relevant(qrel, relevance_level)

    return next(
        (
            1 / rank
            for rank, doc in enumerate(sim_map2ranking(run, k), 1)
            if doc in relevant
        ),
        0.0,
    )


def average_precision(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Mean of the precision values at the ranks of the relevant documents.

    Unretrieved relevant documents contribute zero.
    Averaged over queries, this is MAP (Manning et al. 2008, ch. 8).

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> average_precision({"a": 1, "b": 1}, {"a": 0.9, "c": 0.5, "b": 0.1}, k=None, relevance_level=1)
        0.8333333333333333
    """
    relevant = _relevant(qrel, relevance_level)

    if not relevant:
        return 0.0

    hit_list = np.array(
        [doc in relevant for doc in sim_map2ranking(run, k)], dtype=np.float64
    )
    precisions = np.cumsum(hit_list) / np.arange(1, len(hit_list) + 1)

    return float(np.sum(precisions * hit_list)) / len(relevant)


def _dcg_metric(burges: bool, normalized: bool) -> QueryMetricFunc:
    """Build (n)DCG with graded gains rel / log2(rank + 1).

    Follows Järvelin and Kekäläinen, Cumulated gain-based evaluation of IR techniques, TOIS 2002.
    With burges, the gain is 2^rel - 1 (Burges et al., Learning to rank using gradient descent, ICML 2005).
    The ideal ranking sorts all relevant judgments by grade.
    """

    def metric(
        qrel: Mapping[str, int],
        run: Mapping[str, float],
        k: int | None,
        relevance_level: int,
    ) -> float:
        relevant = _relevant(qrel, relevance_level)
        score = _dcg([relevant.get(doc, 0) for doc in sim_map2ranking(run, k)], burges)

        if not normalized:
            return score

        ideal = sorted(relevant.values(), reverse=True)[:k]

        return score / (_dcg(ideal, burges) or 1.0)

    return metric


def correctness(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Pairwise ranking correctness from -1 (reversed) to 1 (preserved).

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> correctness({"a": 3, "b": 2, "c": 1}, {"a": 0.9, "b": 0.5, "c": 0.1}, k=None, relevance_level=1)
        1.0
    """
    concordant, discordant, _ = concordances(qrel, run, k, relevance_level)

    return (
        (concordant - discordant) / (concordant + discordant)
        if concordant + discordant
        else 0.0
    )


def completeness(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Fraction of ordered relevance pairs that are covered by the top-k results.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> completeness({"a": 3, "b": 2, "c": 1}, {"a": 0.9, "b": 0.5, "c": 0.1}, k=2, relevance_level=1)
        0.3333333333333333
    """
    concordant, discordant, total = concordances(qrel, run, k, relevance_level)

    return (concordant + discordant) / total if total else 0.0


def mean_score(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float:
    """Mean retrieval score of the top-k results, 0 if nothing was retrieved.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> mean_score({"a": 3, "b": 2, "c": 1}, {"a": 0.9, "b": 0.5, "c": 0.1}, k=2, relevance_level=1)
        0.7
        >>> mean_score({"a": 3}, {}, k=2, relevance_level=1)
        0.0
    """
    scores = [run[doc] for doc in sim_map2ranking(run, k)]

    return statistics.mean(scores) if scores else 0.0


def kendall_tau(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
    relevance_level: int,
) -> float | None:
    """Kendall's tau between run and qrel scores of the relevant top-k results.

    Queries with fewer than two such documents, or with constant qrel or run scores among them, are skipped.

    Args:
        qrel: Relevance of the cases for the query, higher is more relevant.
        run: Scores of the retrieved cases for the query.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> qrel = {"a": 3, "b": 2, "c": 1, "d": 0}
        >>> kendall_tau(qrel, {"a": 0.99, "b": 0.66, "c": 0.33, "d": 0.01}, k=None, relevance_level=1)
        1.0
        >>> kendall_tau(qrel, {"a": 0.01, "b": 0.33, "c": 0.66, "d": 0.99}, k=None, relevance_level=1)
        -1.0
        >>> kendall_tau({"a": 1, "b": 1}, {"a": 0.9, "b": 0.1}, k=None, relevance_level=1) is None
        True
    """
    from scipy.stats import kendalltau

    docs = list(_relevant(qrel, relevance_level).keys() & set(sim_map2ranking(run, k)))

    if len(docs) < 2:
        return None

    tau = float(
        kendalltau([qrel[doc] for doc in docs], [run[doc] for doc in docs]).statistic
    )

    return None if math.isnan(tau) else tau


type MetricName = Literal[
    "hits",
    "hit_rate",
    "precision",
    "recall",
    "f1",
    "r_precision",
    "mrr",
    "map",
    "dcg",
    "dcg_burges",
    "ndcg",
    "ndcg_burges",
    "correctness",
    "completeness",
    "mean_score",
    "kendall_tau",
]
"""Names of the built-in metrics, see `Metric`."""

QUERY_METRICS: Mapping[MetricName, QueryMetricFunc] = {
    "hits": hits,
    "hit_rate": hit_rate,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "r_precision": r_precision,
    "mrr": reciprocal_rank,
    "map": average_precision,
    "dcg": _dcg_metric(burges=False, normalized=False),
    "dcg_burges": _dcg_metric(burges=True, normalized=False),
    "ndcg": _dcg_metric(burges=False, normalized=True),
    "ndcg_burges": _dcg_metric(burges=True, normalized=True),
    "correctness": correctness,
    "completeness": completeness,
    "mean_score": mean_score,
    "kendall_tau": kendall_tau,
}


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Metric(EvalMetricFunc):
    """A built-in retrieval metric of one query, which `compute` aggregates over the queries.

    It is undefined (None) for some queries, such as `kendall_tau` with fewer than two relevant cases.
    Its text, such as `precision@5-l2`, names it in reports.

    Args:
        name: The metric.
        k: Number of top-ranked cases that count, all if None.
        relevance_level: Minimal relevance of a relevant case.

    Examples:
        >>> metric = Metric(name="ndcg", k=5, relevance_level=2)
        >>> str(metric), str(Metric(name="precision"))
        ('ndcg@5-l2', 'precision')
        >>> metric({"a": 2, "b": 1}, {"a": 0.9, "b": 0.5})
        1.0
    """

    name: MetricName
    k: Annotated[int, Field(ge=1)] | None = None
    relevance_level: int = 1
    kind: Literal["metric"] = field(default="metric", repr=False)

    @override
    def __str__(self) -> str:
        cutoff = "" if self.k is None else f"@{self.k}"
        level = f"-l{self.relevance_level}" if self.relevance_level != 1 else ""

        return f"{self.name}{cutoff}{level}"

    @override
    def __call__(
        self, qrel: Mapping[str, int], run: Mapping[str, float], /
    ) -> float | None:
        return QUERY_METRICS[self.name](qrel, run, self.k, self.relevance_level)


DEFAULT_METRICS: tuple[Metric, ...] = tuple(
    Metric(name=name)
    for name in (
        "precision",
        "recall",
        "f1",
        "map",
        "ndcg",
        "correctness",
        "completeness",
    )
)
"""The metrics that `compute` uses by default."""


def compute[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    run: QueryCaseMatrix[Q, C, S],
    metrics: Sequence[EvalMetricFunc] = DEFAULT_METRICS,
    aggregation: Callable[[Sequence[float]], float] = statistics.fmean,
) -> dict[EvalMetricFunc, float]:
    """Evaluate retrieval results against relevance judgments.

    Queries and cases are compared by their text, and queries missing in the qrels or the run are skipped.
    The values of a metric over the queries for which it is defined are aggregated, NaN if there is none.

    Args:
        qrels: Relevance of the cases for every query, higher is more relevant.
        run: Similarities of the retrieved cases for every query.
        metrics: Metrics such as `Metric` or custom functions of the `cbrkit.typing.EvalMetricFunc` protocol.
        aggregation: Aggregates the values of a metric over the queries.

    Returns:
        The value of every metric.

    Examples:
        >>> results = compute(
        ...     qrels={"q": {"a": 2, "b": 1, "c": 0}},
        ...     run={"q": {"a": 0.9, "c": 0.5, "b": 0.1}},
        ...     metrics=[Metric(name="precision", k=2), Metric(name="recall"), Metric(name="ndcg", relevance_level=2)],
        ... )
        >>> {str(metric): value for metric, value in results.items()}
        {'precision@2': 0.5, 'recall': 1.0, 'ndcg-l2': 1.0}
    """
    keys = qrels.keys() & run.keys()

    if missing := sorted(str(key) for key in qrels.keys() - keys):
        logger.warning(
            "Ignoring the queries missing in the run: %s", ", ".join(missing)
        )

    if missing := sorted(str(key) for key in run.keys() - keys):
        logger.warning(
            "Ignoring the queries missing in the qrels: %s", ", ".join(missing)
        )

    pairs = [
        (
            {str(case): value for case, value in qrels[query].items()},
            {str(case): float(value) for case, value in run[query].items()},
        )
        for query in keys
    ]
    values: dict[EvalMetricFunc, float] = {}

    for metric in metrics:
        scores = [
            score
            for qrel, ranked in pairs
            if (score := metric(qrel, ranked)) is not None
        ]
        values[metric] = aggregation(scores) if scores else float("nan")

    return values


def generate_metrics(
    names: Iterable[MetricName],
    ks: Iterable[int | None] = (None,),
    relevance_levels: Iterable[int] = (1,),
) -> list[Metric]:
    """Every combination of metric names, cutoffs, and relevance levels.

    Args:
        names: Names of the metrics.
        ks: Numbers of top-ranked cases that count, None for all.
        relevance_levels: Minimal relevance of a relevant case.

    Examples:
        >>> [str(metric) for metric in generate_metrics(["precision", "recall"], ks=[1, 5])]
        ['precision@1', 'precision@5', 'recall@1', 'recall@5']
    """
    return [
        Metric(name=name, k=k, relevance_level=level)
        for name, k, level in itertools.product(names, ks, relevance_levels)
    ]


def similarities_to_qrels[Q, C, S: Float](
    similarities: QueryCaseMatrix[Q, C, S],
    max_qrel: int | None = None,
    min_qrel: int = 1,
    round_mode: Literal["floor", "ceil", "nearest"] = "nearest",
    auto_scale: bool = True,
) -> QueryCaseMatrix[Q, C, int]:
    """Convert float similarity scores to integer relevance judgments.

    Args:
        similarities: Similarities of the cases for every query.
        max_qrel: Largest relevance, which the most similar case gets,
            the cases are ranked from `min_qrel` upwards by their order if None.
        min_qrel: Smallest relevance.
        round_mode: How scaled similarities are rounded to relevances.
        auto_scale: Scale by the smallest and largest similarity instead of between 0 and 1.

    Examples:
        >>> similarities_to_qrels({"q": {"a": 0.9, "b": 0.2, "c": 0.5}})
        {'q': {'b': 1, 'c': 2, 'a': 3}}
        >>> similarities_to_qrels({"q": {"a": 0.9, "b": 0.2, "c": 0.5}}, max_qrel=2)
        {'q': {'a': 2, 'b': 1, 'c': 1}}
        >>> similarities_to_qrels({"q": {}}, max_qrel=2)
        {'q': {}}
    """
    if max_qrel is None:
        return {
            query: {
                case: rank
                for rank, case in enumerate(
                    reversed(sim_map2ranking(case_sims)),
                    start=min_qrel,
                )
            }
            for query, case_sims in similarities.items()
        }

    min_sim, max_sim = 0.0, 1.0

    if auto_scale and (
        values := [
            float(sim) for sims in similarities.values() for sim in sims.values()
        ]
    ):
        min_sim, max_sim = min(values), max(values)

    def relevance(sim: S) -> int:
        normalized = (
            (float(sim) - min_sim) / (max_sim - min_sim) if max_sim != min_sim else 0.0
        )
        scaled = normalized * (max_qrel - min_qrel) + min_qrel

        match round_mode:
            case "floor":
                return math.floor(scaled)
            case "ceil":
                return math.ceil(scaled)
            case "nearest":
                return math.floor(scaled + 0.5)
            case _:
                raise ValueError(f"Invalid rounding mode: {round_mode!r}")

    return {
        query: {case: relevance(sim) for case, sim in case_sims.items()}
        for query, case_sims in similarities.items()
    }
