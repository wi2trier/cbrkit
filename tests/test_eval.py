import pytest

import cbrkit

QRELS = {
    "q1": {"a": 3, "b": 2, "c": 1, "d": 0, "e": 2},
    "q2": {"x": 1, "y": 2},
    "q3": {"m": 1, "n": 3},
}
RUN = {
    "q1": {"a": 0.9, "b": 0.3, "c": 0.8, "d": 0.7, "f": 0.5},
    "q2": {"z": 0.9, "y": 0.8, "w": 0.1},
    "q3": {"o": 0.2, "p": 0.1},
}

# reference values computed with ranx 0.3.21
EXPECTED = {
    "hits": 1.3333333333333333,
    "hits@2-l2": 0.6666666666666666,
    "hit_rate": 0.6666666666666666,
    "hit_rate@2-l2": 0.6666666666666666,
    "precision": 0.3111111111111111,
    "precision@2-l2": 0.3333333333333333,
    "recall": 0.4166666666666667,
    "recall@2-l2": 0.4444444444444444,
    "f1": 0.35555555555555546,
    "f1-l2": 0.3333333333333333,
    "f1@2-l2": 0.35555555555555557,
    "r_precision": 0.3333333333333333,
    "r_precision@2-l2": 0.1111111111111111,
    "mrr": 0.5,
    "mrr@2-l2": 0.5,
    "map": 0.3,
    "map@2-l2": 0.27777777777777773,
    "dcg": 1.888831625061152,
    "dcg@2-l2": 1.4206198357143052,
    "dcg_burges": 3.5614258119964846,
    "dcg_burges@2-l2": 2.9642630869047912,
    "ndcg": 0.41779372825250377,
    "ndcg@2-l2": 0.44494928086853075,
    "ndcg_burges": 0.4445192868935263,
    "ndcg_burges@2-l2": 0.47269478552080973,
}


METRICS = cbrkit.eval.generate_metrics(
    cbrkit.eval.common.QUERY_METRICS, ks=[None, 2], relevance_levels=[1, 2]
)


@pytest.mark.parametrize("metric", [m for m in METRICS if str(m) in EXPECTED], ids=str)
def test_metric(metric: cbrkit.eval.Metric) -> None:
    result = cbrkit.eval.compute(QRELS, RUN, [metric])

    assert result[metric] == pytest.approx(EXPECTED[str(metric)])


def test_custom_metric_and_aggregation() -> None:
    def found(qrel, run) -> float:
        return float(len(qrel.keys() & run.keys()))

    assert cbrkit.eval.compute(QRELS, RUN, [found], aggregation=sum) == {found: 5.0}
