"""The column paths of the measures, the aggregator and AttributeValue equal their scalar reference."""

import random
from datetime import date, timedelta
from typing import Any

import numpy as np
import pytest

import cbrkit
from cbrkit.sim import pooling
from cbrkit.typing import ColumnSimFunc

random.seed(0)
NUMBERS = [random.choice([0, 5, 10, 50, 99.5, 100, 150]) for _ in range(200)]
WORDS = [
    random.choice(["Kitten", "sitting", "kit", "", " New  York", "new york"])
    for _ in range(200)
]
DATES = [date(2024, 1, 1) + timedelta(days=random.randint(0, 30)) for _ in range(200)]

MEASURES: list[tuple[ColumnSimFunc[Any], list[Any]]] = [
    (cbrkit.sim.numbers.Linear(max_distance=50, min_distance=5), NUMBERS),
    (cbrkit.sim.numbers.Linear(max_distance=10, min_distance=10), NUMBERS),
    (cbrkit.sim.numbers.LinearInterval(lower=0, upper=100), NUMBERS),
    (cbrkit.sim.numbers.Threshold(value=5), NUMBERS),
    (cbrkit.sim.numbers.Exponential(alpha=0.1), NUMBERS),
    (cbrkit.sim.numbers.Sigmoid(alpha=2, theta=10), NUMBERS),
    (
        cbrkit.sim.dates.DateLinear(
            max_distance=timedelta(days=10), min_distance=timedelta(days=2)
        ),
        DATES,
    ),
    (cbrkit.sim.strings.Levenshtein(case_sensitive=False), WORDS),
    (cbrkit.sim.strings.Indel(score_cutoff=0.5), WORDS),
    (cbrkit.sim.strings.Jaro(), WORDS),
    (cbrkit.sim.strings.JaroWinkler(prefix_weight=0.2, case_sensitive=False), WORDS),
    (cbrkit.sim.generic.Equality(), WORDS),
    (
        cbrkit.sim.generic.Equality(case_sensitive=False, normalize_whitespace=True),
        WORDS,
    ),
    (cbrkit.sim.generic.TypeEquality(), [*WORDS[:100], *NUMBERS[:100]]),
    (cbrkit.sim.generic.Static(value=0.5), WORDS),
    (
        cbrkit.sim.generic.Table(
            entries=[("kitten", "sitting", 0.5), ("kit", "", 0.2)],
            default=0.0,
            case_sensitive=False,
        ),
        WORDS,
    ),
]


@pytest.mark.parametrize(("measure", "values"), MEASURES)
def test_measure_columns(measure: ColumnSimFunc[Any], values: list[Any]):
    queries = values[::-1]
    expected = [
        measure(case, query) for case, query in zip(values, queries, strict=True)
    ]

    assert measure.columns(values, queries) == pytest.approx(expected)


@pytest.mark.parametrize(
    "func",
    [
        pooling.Mean(),
        pooling.GeometricMean(),
        pooling.HarmonicMean(),
        pooling.Min(),
        pooling.Max(),
        pooling.Sum(),
        pooling.Median(),
        pooling.KMax(k=2),
    ],
)
@pytest.mark.parametrize("weights", [{}, {"a": 2.0, "b": 0.5, "c": 0.0}])
def test_aggregator_columns(
    func: cbrkit.typing.PoolingFunc[float], weights: dict[str, float]
):
    agg = cbrkit.sim.Aggregator(pooling=func, weights=weights, default_weight=1.5)
    keys = ["a", "b", "c", "d"]
    rng = np.random.default_rng(0)
    sims = rng.choice([0.0, 0.25, 0.5, 1.0], size=(100, len(keys)))
    present = rng.random((100, len(keys))) > 0.3
    expected = [
        agg({key: sim for key, sim, keep in zip(keys, row, mask, strict=True) if keep})
        for row, mask in zip(sims.tolist(), present.tolist(), strict=True)
    ]

    assert agg.columns(keys, sims, present) == pytest.approx(expected)


def _pair_func(measure: ColumnSimFunc[Any]) -> cbrkit.typing.SimFunc[Any, float]:
    return lambda case, query: measure(case, query)


@pytest.mark.parametrize("skip_missing_query", [False, True])
def test_attribute_value_columns(skip_missing_query: bool):
    measures = {
        "price": cbrkit.sim.numbers.Linear(max_distance=100),
        "name": cbrkit.sim.strings.Levenshtein(),
        "color": cbrkit.sim.generic.Equality(),
    }
    agg = cbrkit.sim.Aggregator(weights={"price": 2.0})

    def entry(idx: int) -> dict[str, Any]:
        return {
            "price": NUMBERS[idx],
            "name": WORDS[idx],
            **({"color": WORDS[idx][:1]} if idx % 3 else {}),
        }

    pairs = [(entry(idx), entry(199 - idx)) for idx in range(200)]
    vectorised = cbrkit.sim.AttributeValue(
        attributes=measures,
        aggregator=agg,
        default=0.0,
        skip_missing_query=skip_missing_query,
    )
    # plain functions take the scalar path
    scalar = cbrkit.sim.AttributeValue(
        attributes={name: _pair_func(measure) for name, measure in measures.items()},
        aggregator=agg,
        default=0.0,
        skip_missing_query=skip_missing_query,
    )

    for actual, expected in zip(vectorised(pairs), scalar(pairs), strict=True):
        assert actual.value == pytest.approx(expected.value)
        assert actual.similarities == pytest.approx(expected.similarities)


@pytest.mark.parametrize("metric", [cbrkit.sim.embed.Cosine(), cbrkit.sim.embed.Dot()])
def test_embedding_matrix_equals_scalar(metric: cbrkit.typing.MatrixSimFunc) -> None:
    rng = np.random.default_rng(0)
    cases = rng.normal(size=(20, 8))
    queries = rng.normal(size=(3, 8))
    cases[0] = 0.0
    queries[1] = 0.0

    expected = [[metric(case, query) for query in queries] for case in cases]

    np.testing.assert_allclose(metric.matrix(cases, queries), expected)
