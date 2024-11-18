import statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence

from ..helpers import unpack_sim
from ..typing import Float

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

    correctness = (
        (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)
        if (concordant_pairs + discordant_pairs) > 0
        else 0.0
    )

    completeness = (
        (concordant_pairs + discordant_pairs) / total_pairs if total_pairs > 0 else 0.0
    )

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
