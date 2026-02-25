import itertools
import statistics
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal, cast

from ..helpers import (
    get_logger,
    normalize_and_scale,
    round,
    sim_map2ranking,
    unpack_float,
    unpack_floats,
)
from ..typing import ConversionFunc, EvalMetricFunc, Float, QueryCaseMatrix

logger = get_logger(__name__)


def compute_score_metric[C, S: Float, T](
    qrel_scores: Mapping[C, S],
    run_scores: Mapping[C, S],
    metric_func: Callable[[list[float], list[float]], T],
) -> T:
    keys = set(qrel_scores.keys()).intersection(set(run_scores.keys()))

    return metric_func(
        unpack_floats(qrel_scores[key] for key in keys),
        unpack_floats(run_scores[key] for key in keys),
    )


def compute_score_metrics[Q, C, S: Float, T](
    qrel_scores: QueryCaseMatrix[Q, C, S],
    run_scores: QueryCaseMatrix[Q, C, S],
    metric_funcs: dict[str, Callable[[list[float], list[float]], T]],
    aggregation_func: ConversionFunc[list[T], T] | None = None,
) -> dict[str, T | float]:
    metric_values: dict[str, T | float] = {}
    keys = set(qrel_scores.keys()).intersection(set(run_scores.keys()))

    for metric_name, metric_func in metric_funcs.items():
        scores = [
            compute_score_metric(qrel_scores[key], run_scores[key], metric_func)
            for key in keys
        ]

        if len(scores) == 0:
            metric_value = float("nan")
        elif aggregation_func is None:
            metric_value = statistics.mean(cast(list[float], scores))
        else:
            metric_value = aggregation_func(scores)

        metric_values[metric_name] = metric_value

    return metric_values


def parse_metric(spec: str) -> tuple[str, int, int]:
    """Parse a metric specification string into its components.

    Args:
        spec: Metric string, optionally with `@k` and `-lN` suffixes.

    Returns:
        A tuple of (metric_name, k, relevance_level).

    Examples:
        >>> parse_metric("precision")
        ('precision', 0, 1)
        >>> parse_metric("precision@5")
        ('precision', 5, 1)
        >>> parse_metric("ndcg-l2")
        ('ndcg', 0, 2)
        >>> parse_metric("recall@10-l3")
        ('recall', 10, 3)
    """
    rel_lvl = 1
    k = 0

    if "-l" in spec:
        spec, rel_lvl = spec.split("-l")

    if "@" in spec:
        spec, k = spec.split("@")

    return spec, int(k), int(rel_lvl)


def concordances(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int,
    relevance_level: int,
) -> tuple[int, int, int]:
    """Compute the number of concordant and discordant pairs in a ranking.

    Args:
        qrel: The query relevance judgments.
        run: The ranking produced by the system.
        k: The number of top documents to consider.
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
        >>> concordances(qrel, run1, 0, 1)
        (9, 1, 10)
        >>> run2 = {
        ...     "case1": 0.6,
        ...     "case2": 0.4,
        ...     "case3": 0.55,
        ...     "case4": 0.7,
        ...     "case5": 0.58,
        ... }
        >>> concordances(qrel, run2, 0, 1)
        (5, 5, 10)
    """
    # We only inverse the similarities to compute the ranking and get the top k
    # Later, we us the original similarities as their ordering corresponds to the qrel odering
    sorted_run = sorted(run.items(), key=lambda x: x[1], reverse=True)
    run_k = dict(sorted_run[: k if k > 0 else len(sorted_run)])
    qrel_relevant = {
        doc: score for doc, score in qrel.items() if score >= relevance_level
    }

    keys = list(qrel_relevant.keys())

    concordant_pairs = 0
    discordant_pairs = 0
    total_pairs = 0

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            idx1, idx2 = keys[i], keys[j]
            qrel1, qrel2 = qrel[idx1], qrel[idx2]

            if qrel1 != qrel2:
                total_pairs += 1

                if idx1 in run_k and idx2 in run_k:
                    run1, run2 = run[idx1], run[idx2]

                    if (qrel1 < qrel2 and run1 < run2) or (
                        qrel1 > qrel2 and run1 > run2
                    ):
                        concordant_pairs += 1
                    else:
                        discordant_pairs += 1

    return concordant_pairs, discordant_pairs, total_pairs


def _correctness_single(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int,
    relevance_level: int,
) -> float:
    concordant_pairs, discordant_pairs, total_pairs = concordances(
        qrel, run, k, relevance_level
    )

    return (
        (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)
        if (concordant_pairs + discordant_pairs) > 0
        else 0.0
    )


def correctness(
    qrels: QueryCaseMatrix[str, str, int],
    run: QueryCaseMatrix[str, str, float],
    k: int,
    relevance_level: int,
) -> float:
    """Compute average pairwise ranking correctness over shared queries.

    Examples:
        >>> qrels = {"q": {"a": 3, "b": 2, "c": 1}}
        >>> run = {"q": {"a": 0.9, "b": 0.5, "c": 0.1}}
        >>> correctness(qrels, run, k=0, relevance_level=1)
        1.0
    """
    keys = set(qrels.keys()).intersection(set(run.keys()))
    scores = [
        _correctness_single(qrels[key], run[key], k, relevance_level) for key in keys
    ]

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


def _completeness_single(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int,
    relevance_level: int,
) -> float:
    concordant_pairs, discordant_pairs, total_pairs = concordances(
        qrel, run, k, relevance_level
    )

    return (
        (concordant_pairs + discordant_pairs) / total_pairs if total_pairs > 0 else 0.0
    )


def completeness(
    qrels: QueryCaseMatrix[str, str, int],
    run: QueryCaseMatrix[str, str, float],
    k: int,
    relevance_level: int,
) -> float:
    """Compute average pairwise ranking coverage over shared queries.

    Examples:
        >>> qrels = {"q": {"a": 3, "b": 2, "c": 1}}
        >>> run = {"q": {"a": 0.9, "b": 0.5, "c": 0.1}}
        >>> completeness(qrels, run, k=2, relevance_level=1)
        0.3333333333333333
    """
    keys = set(qrels.keys()).intersection(set(run.keys()))
    scores = [
        _completeness_single(qrels[key], run[key], k, relevance_level) for key in keys
    ]

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


def mean_score(
    qrels: QueryCaseMatrix[str, str, int],
    run: QueryCaseMatrix[str, str, float],
    k: int,
    relevance_level: int,
) -> float:
    """Compute mean retrieval score over the top-k results for each query.

    Examples:
        >>> qrels = {"q": {"a": 3, "b": 2, "c": 1}}
        >>> run = {"q": {"a": 0.9, "b": 0.5, "c": 0.1}}
        >>> mean_score(qrels, run, k=2, relevance_level=1)
        0.7
    """
    keys = set(qrels.keys()).intersection(set(run.keys()))

    scores: list[float] = []

    for key in keys:
        sorted_run = sorted(run[key].values(), reverse=True)
        run_k = sorted_run[: k if k > 0 else len(sorted_run)]

        scores.append(statistics.mean(run_k))

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


def kendall_tau(
    qrels: QueryCaseMatrix[str, str, int],
    run: QueryCaseMatrix[str, str, float],
    k: int,
    relevance_level: int,
) -> float:
    """Compute average Kendall's tau between run and qrel-induced rankings.

    For each shared query, only documents with relevance >= relevance_level
    that also appear in the (optionally top-k filtered) run are considered.
    Scipy's kendalltau is called directly on the paired qrel/run scores,
    which naturally handles ties.

    Examples:
        >>> qrels = {"q": {"a": 3, "b": 2, "c": 1, "d": 0}}
        >>> run_perfect = {"q": {"a": 0.99, "b": 0.66, "c": 0.33, "d": 0.01}}
        >>> run_reverse = {"q": {"a": 0.01, "b": 0.33, "c": 0.66, "d": 0.99}}
        >>> kendall_tau(qrels, run_perfect, k=0, relevance_level=1)
        1.0
        >>> kendall_tau(qrels, run_reverse, k=0, relevance_level=1)
        -1.0
    """
    from scipy.stats import kendalltau

    keys = set(qrels.keys()).intersection(set(run.keys()))

    scores: list[float] = []

    for key in keys:
        qrel_relevant = {
            doc for doc, score in qrels[key].items() if score >= relevance_level
        }

        sorted_run = sorted(run[key].keys(), key=lambda x: run[key][x], reverse=True)
        run_k = set(sorted_run[: k if k > 0 else len(sorted_run)])

        common_docs = list(qrel_relevant & run_k)

        if len(common_docs) < 2:
            continue

        qrel_scores = [qrels[key][doc] for doc in common_docs]
        run_scores = [run[key][doc] for doc in common_docs]
        score = kendalltau(qrel_scores, run_scores)

        scores.append(float(score.statistic))

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


DEFAULT_METRICS = (
    "precision",
    "recall",
    "f1",
    "map",
    "ndcg",
    "correctness",
    "completeness",
)

CBRKIT_METRICS: dict[str, EvalMetricFunc] = {
    "correctness": correctness,
    "completeness": completeness,
    "mean_score": mean_score,
    "kendall_tau": kendall_tau,
}


def compute[Q, C, S: Float](
    qrels: QueryCaseMatrix[Q, C, int],
    run: QueryCaseMatrix[Q, C, S],
    metrics: Sequence[str] = DEFAULT_METRICS,
    metric_funcs: Mapping[str, EvalMetricFunc] | None = None,
) -> dict[str, float]:
    import ranx

    if metric_funcs is None:
        metric_funcs = {}

    all_custom_metric_names = set(metric_funcs.keys()).union(CBRKIT_METRICS.keys())

    ranx_metrics = [
        x for x in metrics if not any(x.startswith(y) for y in all_custom_metric_names)
    ]
    custom_metrics = [x for x in metrics if x not in ranx_metrics]

    keys = set(qrels.keys()).intersection(set(run.keys()))
    missing_qrel_keys = set(run.keys()).difference(keys)
    missing_run_keys = set(qrels.keys()).difference(keys)

    if len(missing_run_keys) > 0:
        sorted_keys = sorted([str(x) for x in missing_run_keys])
        logger.warning(
            f"Missing keys in qrels: {', '.join(sorted_keys)}. "
            "Ignoring these keys for evaluation."
        )

    if len(missing_qrel_keys) > 0:
        sorted_keys = sorted([str(x) for x in missing_qrel_keys])
        logger.warning(
            f"Missing keys in run: {', '.join(sorted_keys)}. "
            "Ignoring these keys for evaluation."
        )

    parsed_qrels = {
        str(qk): {str(ck): cv for ck, cv in qv.items()}
        for qk, qv in qrels.items()
        if qk in keys
    }
    parsed_run = {
        str(qk): {str(ck): unpack_float(cv) for ck, cv in qv.items()}
        for qk, qv in run.items()
        if qk in keys
    }

    eval_results: dict[str, float] = {}

    if ranx_metrics:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            ranx_results = ranx.evaluate(
                ranx.Qrels(parsed_qrels),
                ranx.Run(parsed_run),
                metrics=ranx_metrics,
            )

        if isinstance(ranx_results, dict):
            eval_results.update(ranx_results)
        else:
            assert len(ranx_metrics) == 1
            eval_results = {ranx_metrics[0]: ranx_results}

    for metric_spec in custom_metrics:
        metric, k, relevance_level = parse_metric(metric_spec)

        if metric in CBRKIT_METRICS:
            eval_results[metric_spec] = CBRKIT_METRICS[metric](
                parsed_qrels, parsed_run, k, relevance_level
            )
        elif metric in metric_funcs:
            eval_results[metric_spec] = metric_funcs[metric](
                parsed_qrels, parsed_run, k, relevance_level
            )

    return eval_results


def generate_metric(
    metric: str,
    k: int | None = None,
    relevance_level: int | None = None,
) -> str:
    """Generate a metric specification string from components.

    Args:
        metric: The base metric name.
        k: Optional cutoff value.
        relevance_level: Optional relevance level.

    Returns:
        A formatted metric string.

    Examples:
        >>> generate_metric("precision")
        'precision'
        >>> generate_metric("precision", k=5)
        'precision@5'
        >>> generate_metric("ndcg", relevance_level=2)
        'ndcg-l2'
        >>> generate_metric("recall", k=10, relevance_level=3)
        'recall@10-l3'
    """
    if k is None and relevance_level is None:
        return metric
    elif k is None:
        return f"{metric}-l{relevance_level}"
    elif relevance_level is None:
        return f"{metric}@{k}"

    return f"{metric}@{k}-l{relevance_level}"


def generate_metrics(
    metrics: Iterable[str] = DEFAULT_METRICS,
    ks: Iterable[int | None] | int | None = None,
    relevance_levels: Iterable[int | None] | int | None = None,
) -> list[str]:
    """Generate metric specification strings for multiple cutoff points.

    Args:
        metrics: Base metric names.
        ks: Cutoff values (single or iterable).
        relevance_levels: Relevance levels (single or iterable).

    Returns:
        A list of formatted metric strings.

    Examples:
        >>> generate_metrics(["precision", "recall"], ks=5)
        ['precision@5', 'recall@5']
    """
    if not isinstance(ks, Iterable):
        ks = [ks]

    if not isinstance(relevance_levels, Iterable):
        relevance_levels = [relevance_levels]

    return [
        generate_metric(*args)
        for args in itertools.product(metrics, ks, relevance_levels)
    ]


def similarities_to_qrels[Q, C](
    similarities: QueryCaseMatrix[Q, C, float],
    max_qrel: int | None = None,
    min_qrel: int = 1,
    round_mode: Literal["floor", "ceil", "nearest"] = "nearest",
    auto_scale: bool = True,
) -> QueryCaseMatrix[Q, C, int]:
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

    if auto_scale:
        min_sim = min(min(entries.values()) for entries in similarities.values())
        max_sim = max(max(entries.values()) for entries in similarities.values())
    else:
        min_sim = 0.0
        max_sim = 1.0

    return {
        query: {
            case: round(
                normalize_and_scale(sim, min_sim, max_sim, min_qrel, max_qrel),
                round_mode,
            )
            for case, sim in case_sims.items()
        }
        for query, case_sims in similarities.items()
    }
