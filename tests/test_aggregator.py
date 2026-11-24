"""Tests for similarity aggregator functionality."""

import math
import statistics

import pytest
from pydantic import RootModel

import cbrkit
from cbrkit.sim import pooling


def test_default_aggregator():
    """Test default aggregator with mean pooling."""
    agg = cbrkit.sim.Aggregator()
    assert agg.pooling == pooling.Mean()
    assert agg.weights == {}
    assert agg.default_weight == 1.0

    # Test with sequence and mapping
    assert agg([0.5, 0.75, 1.0]) == pytest.approx(0.75)
    assert agg({"a": 0.5, "b": 0.75, "c": 1.0}) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("func", "reference"),
    [
        (pooling.Mean(), statistics.fmean),
        (pooling.GeometricMean(), statistics.geometric_mean),
        (pooling.HarmonicMean(), statistics.harmonic_mean),
        (pooling.Median(), statistics.median),
        (pooling.MedianLow(), statistics.median_low),
        (pooling.MedianHigh(), statistics.median_high),
        (pooling.Min(), min),
        (pooling.Max(), max),
        (pooling.Sum(), sum),
    ],
)
def test_builtin_pooling_functions(func, reference):
    """Test various built-in pooling functions."""
    similarities = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg = cbrkit.sim.Aggregator[int](pooling=func)
    expected = reference(similarities)

    # Test with sequence
    result = agg(similarities)
    assert result == pytest.approx(expected)

    # Test with mapping
    sim_map = {i: v for i, v in enumerate(similarities)}
    result = agg(sim_map)
    assert result == pytest.approx(expected)


def test_mode_pooling():
    """Test mode pooling function specifically."""
    similarities = [0.5, 0.5, 0.8, 0.8, 0.8]
    agg = cbrkit.sim.Aggregator(pooling=pooling.Mode())
    assert agg(similarities) == pytest.approx(0.8)


def test_weighted_aggregation():
    """Test aggregation with weights."""
    similarities = [0.4, 0.6, 0.8]
    weights = {0: 1, 1: 2, 2: 1}

    # weights by position
    agg = cbrkit.sim.Aggregator(weights=weights)
    result = agg(similarities)
    expected = (0.4 * 1 + 0.6 * 2 + 0.8 * 1) / sum(weights.values())
    assert result == pytest.approx(expected)

    # Mapping weights
    sim_map = {"a": 0.4, "b": 0.6, "c": 0.8}
    weight_map = {"a": 1, "b": 2, "c": 1}
    agg = cbrkit.sim.Aggregator(weights=weight_map)
    assert agg(sim_map) == pytest.approx(expected)

    # Default weight for missing keys
    weight_map_partial = {"a": 2, "b": 3}  # "c" missing
    agg = cbrkit.sim.Aggregator(weights=weight_map_partial, default_weight=1)
    result = agg(sim_map)
    expected = (0.4 * 2 + 0.6 * 3 + 0.8 * 1) / (2 + 3 + 1)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "func",
    [pooling.Min(), pooling.Max(), pooling.Median(), pooling.MedianLow()],
)
def test_weighted_order_statistics(func):
    """Weights must not push order statistics outside the similarity range."""
    similarities = [0.5, 1.0, 0.0]
    weights = {0: 1, 1: 0.5, 2: 0.25}
    agg = cbrkit.sim.Aggregator(pooling=func, weights=weights)
    assert agg(similarities) == pytest.approx(0.5)

    # uniform weights equal the unweighted pooling
    uniform = cbrkit.sim.Aggregator(pooling=func, weights={0: 3, 1: 3, 2: 3})
    assert uniform(similarities) == pytest.approx(func(similarities))


def test_custom_pooling_function():
    """Test with custom pooling function."""

    def variance_pooling(values):
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    similarities = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg = cbrkit.sim.Aggregator(pooling=variance_pooling)

    expected = variance_pooling(similarities)
    assert agg(similarities) == pytest.approx(expected)


def test_edge_cases():
    """Test edge cases."""
    # Single value
    agg = cbrkit.sim.Aggregator()
    assert agg([0.5]) == pytest.approx(0.5)
    assert agg({"a": 0.5}) == pytest.approx(0.5)

    # Empty or zero weights
    assert cbrkit.sim.Aggregator()([]) == 0.0
    agg = cbrkit.sim.Aggregator(default_weight=0)
    assert agg([0.5, 0.6]) == 0.0


def test_weights_by_position():
    """Sequences of similarities are weighted by position."""
    agg = cbrkit.sim.Aggregator(weights={0: 1, 1: 3})
    assert agg([0.0, 1.0]) == pytest.approx(0.75)
    assert agg([0.0, 1.0, 1.0]) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "k,expected",
    [
        (1, 0.1),  # minimum
        (2, 0.3),  # 2nd smallest
        (5, 0.9),  # maximum
        (10, 0.9),  # k > len (returns max)
    ],
)
def test_k_min_pooling(k, expected):
    """Test k-min pooling function."""
    similarities = [0.1, 0.3, 0.5, 0.7, 0.9]
    agg = cbrkit.sim.Aggregator(pooling=pooling.KMin(k=k))
    assert agg(similarities) == pytest.approx(expected)


@pytest.mark.parametrize(
    "k,expected",
    [
        (1, 0.9),  # maximum
        (2, 0.7),  # 2nd largest
        (5, 0.1),  # minimum
        (10, 0.1),  # k > len (returns min)
    ],
)
def test_k_max_pooling(k, expected):
    """Test k-max pooling function."""
    similarities = [0.1, 0.3, 0.5, 0.7, 0.9]
    agg = cbrkit.sim.Aggregator(pooling=pooling.KMax(k=k))
    assert agg(similarities) == pytest.approx(expected)


@pytest.mark.parametrize(
    "p,formula",
    [
        (1, lambda sims: sum(sims)),  # Manhattan
        (2, lambda sims: math.sqrt(sum(s**2 for s in sims))),  # Euclidean
        (3, lambda sims: sum(s**3 for s in sims) ** (1 / 3)),  # p=3
    ],
)
def test_minkowski_pooling(p, formula):
    """Test Minkowski pooling function."""
    similarities = [0.4, 0.6, 0.8]
    agg = cbrkit.sim.Aggregator(pooling=pooling.Minkowski(p=p))
    assert agg(similarities) == pytest.approx(formula(similarities))


class SerializedAggregator(RootModel[cbrkit.sim.Aggregator[str]]):
    """An aggregator in its serialized form."""


def test_pooling_kinds_serialize():
    agg = cbrkit.sim.Aggregator(pooling=pooling.KMax(k=2))
    data = SerializedAggregator(agg).model_dump(mode="json")

    assert data["pooling"] == {"k": 2, "kind": "k_max"}
    assert SerializedAggregator.model_validate(data).root == agg


def test_custom_pooling_edge_cases():
    """Test edge cases for custom pooling functions."""
    with pytest.raises(ValueError):
        pooling.KMin(k=0)

    # Single element
    assert cbrkit.sim.Aggregator(pooling=pooling.KMin(k=1))([0.5]) == pytest.approx(0.5)
    assert cbrkit.sim.Aggregator(pooling=pooling.KMax(k=1))([0.5]) == pytest.approx(0.5)
