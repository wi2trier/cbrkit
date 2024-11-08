import itertools
import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence

from cbrkit.helpers import unpack_sim
from cbrkit.typing import Float

# https://amenra.github.io/ranx/metrics/

__all__ = ["compute", "correctness_completeness", "metrics_at_k", "parse_metric"]


def correctness_completeness(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int | None = None,
) -> tuple[float, float]:
    keys = set(qrels.keys()).intersection(set(run.keys()))

    scores = [_correctness_completeness_single(qrels[key], run[key], k) for key in keys]
    correctness_scores = [score[0] for score in scores]
    completeness_scores = [score[1] for score in scores]

    try:
        return statistics.mean(correctness_scores), statistics.mean(completeness_scores)
    except statistics.StatisticsError:
        return float("nan"), float("nan")


def _correctness_completeness_single(
    qrel: Mapping[str, int],
    run: Mapping[str, float],
    k: int | None,
) -> tuple[float, float]:
    sorted_run = sorted(run.items(), key=lambda x: x[1], reverse=True)
    run_k = {x[0]: x[1] for x in sorted_run[:k]}

    orders = 0
    concordances = 0
    disconcordances = 0

    correctness = 1
    completeness = 1

    for (idx1, qrel1), (idx2, qrel2) in itertools.product(qrel.items(), qrel.items()):
        if idx1 != idx2 and qrel1 > qrel2:
            orders += 1

            run1 = run_k.get(idx1)
            run2 = run_k.get(idx2)

            if run1 is not None and run2 is not None:
                if run1 > run2:
                    concordances += 1
                elif run1 < run2:
                    disconcordances += 1

    if concordances + disconcordances > 0:
        correctness = (concordances - disconcordances) / (
            concordances + disconcordances
        )
    if orders > 0:
        completeness = (concordances + disconcordances) / orders

    return correctness, completeness


def metrics_at_k(
    metrics: Iterable[str],
    ks: Iterable[int],
) -> list[str]:
    return [f"{metric}@{k}" for metric in metrics for k in ks]


def parse_metric(metric: str) -> tuple[str, int | None]:
    if "@" in metric:
        metric_name, k = metric.split("@")
        return metric_name, int(k)

    return metric, None


DEFAULT_METRICS = (
    "precision",
    "recall",
    "f1",
    "map",
    "ndcg",
    "correctness_completeness",
)

CUSTOM_METRICS = ("correctness_completeness",)


def compute[QK, CK, S: Float](
    qrels: Mapping[QK, Mapping[CK, int]],
    run: Mapping[QK, Mapping[CK, S]],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    import ranx

    ranx_metrics = [
        x for x in metrics if not any(x.startswith(y) for y in CUSTOM_METRICS)
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

    for custom_metric in custom_metrics:
        metric, k = parse_metric(custom_metric)

        if metric == "correctness_completeness":
            eval_results["correctness"], eval_results["completeness"] = (
                correctness_completeness(parsed_qrels, parsed_run, k)
            )

    return eval_results
