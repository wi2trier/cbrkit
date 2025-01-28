import itertools
import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence

from ..helpers import unpack_float
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix

# https://amenra.github.io/ranx/metrics/


def parse_metric(spec: str) -> tuple[str, int, int]:
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
    qrel_relevant = {k: v for k, v in qrel.items() if v >= relevance_level}

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
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
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
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
    keys = set(qrels.keys()).intersection(set(run.keys()))
    scores = [
        _completeness_single(qrels[key], run[key], k, relevance_level) for key in keys
    ]

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


def mean_score(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
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
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
    from scipy.stats import kendalltau

    keys = set(qrels.keys()).intersection(set(run.keys()))

    scores: list[float] = []

    for key in keys:
        qrel_relevant = {k for k, v in qrels[key].items() if v >= relevance_level}
        sorted_qrel_relevant = sorted(qrel_relevant, key=lambda x: qrels[key][x])

        sorted_run = sorted(run.keys(), key=lambda x: run[key][x], reverse=True)
        run_k = sorted_run[: k if k > 0 else len(sorted_run)]

        max_idx = min(len(run_k), len(sorted_qrel_relevant))
        run_ranking = sorted_run[:max_idx]
        qrel_ranking = sorted_qrel_relevant[:max_idx]

        scores.append(kendalltau(run_ranking, qrel_ranking).statistic)

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

    parsed_qrels = {
        str(qk): {str(ck): cv for ck, cv in qv.items()} for qk, qv in qrels.items()
    }
    parsed_run = {
        str(qk): {str(ck): unpack_float(cv) for ck, cv in qv.items()}
        for qk, qv in run.items()
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        eval_results = ranx.evaluate(
            ranx.Qrels(parsed_qrels),
            ranx.Run(parsed_run),
            metrics=ranx_metrics,
        )

    assert isinstance(eval_results, dict)

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
    if not isinstance(ks, Iterable):
        ks = [ks]

    if not isinstance(relevance_levels, Iterable):
        relevance_levels = [relevance_levels]

    return [
        generate_metric(*args)
        for args in itertools.product(metrics, ks, relevance_levels)
    ]
