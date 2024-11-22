import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import override

from ..helpers import unpack_sim
from ..typing import EvalMetricFunc, Float, QueryCaseMatrix

# https://amenra.github.io/ranx/metrics/

__all__ = [
    "compute",
    "generate_metrics",
    "parse_metric",
]


def parse_metric(spec: str) -> tuple[str, int, int]:
    rel_lvl = 1

    if "-l" in spec:
        spec, rel_lvl = spec.split("-l")

    metric_split = spec.split("@")
    m = metric_split[0]
    k = int(metric_split[1]) if len(metric_split) > 1 else 0

    return m, k, int(rel_lvl)


def generate_metrics(
    metrics: Iterable[str],
    ks: Iterable[int],
) -> list[str]:
    return [f"{metric}@{k}" for metric in metrics for k in ks]


def concordances(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int,
) -> tuple[int, int, int]:
    sorted_run = sorted(run.items(), key=lambda x: x[1], reverse=True)
    run_k = {x[0]: x[1] for x in sorted_run[: k if k > 0 else len(sorted_run)]}

    concordant_pairs = 0
    discordant_pairs = 0
    total_pairs = 0

    case_keys = list(qrel.keys())

    for i in range(len(case_keys)):
        for j in range(i + 1, len(case_keys)):
            idx1, idx2 = case_keys[i], case_keys[j]
            qrel1, qrel2 = qrel[idx1], qrel[idx2]

            if qrel1 != qrel2:
                total_pairs += 1

                if idx1 in run_k and idx2 in run_k:
                    run1, run2 = run_k[idx1], run_k[idx2]

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
) -> float:
    concordant_pairs, discordant_pairs, total_pairs = concordances(qrel, run, k)

    return (
        (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)
        if (concordant_pairs + discordant_pairs) > 0
        else 0.0
    )


@override
def correctness(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
    keys = set(qrels.keys()).intersection(set(run.keys()))
    scores = [_correctness_single(qrels[key], run[key], k) for key in keys]

    try:
        return statistics.mean(scores)
    except statistics.StatisticsError:
        return float("nan")


def _completeness_single(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int,
) -> float:
    concordant_pairs, discordant_pairs, total_pairs = concordances(qrel, run, k)

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
    scores = [_completeness_single(qrels[key], run[key], k) for key in keys]

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
        str(qk): {str(ck): unpack_sim(cv) for ck, cv in qv.items()}
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
