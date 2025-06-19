"""Tests for similarity aggregator functionality."""

import math

import pytest

import cbrkit


def test_default_aggregator():
    """Test default aggregator with mean pooling."""
    agg = cbrkit.sim.aggregator()
    assert agg.pooling == "mean"
    assert agg.pooling_weights is None
    assert agg.default_pooling_weight == 1.0

    # Test with sequence and mapping
    assert agg([0.5, 0.75, 1.0]) == pytest.approx(0.75)
    assert agg({"a": 0.5, "b": 0.75, "c": 1.0}) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "pooling_name",
    [
        "mean",
        "fmean",
        "geometric_mean",
        "harmonic_mean",
        "median",
        "median_low",
        "median_high",
        "min",
        "max",
        "sum",
    ],
)
def test_builtin_pooling_functions(pooling_name):
    """Test various built-in pooling functions."""
    similarities = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg = cbrkit.sim.aggregator(pooling=pooling_name)

    func = cbrkit.sim.pooling.pooling_funcs[pooling_name]
    expected = func(similarities)

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
    agg = cbrkit.sim.aggregator(pooling="mode")
    assert agg(similarities) == pytest.approx(0.8)


def test_weighted_aggregation():
    """Test aggregation with weights."""
    similarities = [0.4, 0.6, 0.8]
    weights = [1, 2, 1]

    # Sequence weights
    agg = cbrkit.sim.aggregator(pooling="mean", pooling_weights=weights)
    result = agg(similarities)
    # Weighted mean with pooling factor: sum(w_i * s_i) / sum(w_i) * len(s) / sum(w_i) * sum(w_i)
    expected = (0.4 * 1 + 0.6 * 2 + 0.8 * 1) / sum(weights)
    assert result == pytest.approx(expected)

    # Mapping weights
    sim_map = {"a": 0.4, "b": 0.6, "c": 0.8}
    weight_map = {"a": 1, "b": 2, "c": 1}
    agg = cbrkit.sim.aggregator(pooling="mean", pooling_weights=weight_map)
    assert agg(sim_map) == pytest.approx(expected)

    # Default weight for missing keys
    weight_map_partial = {"a": 2, "b": 3}  # "c" missing
    agg = cbrkit.sim.aggregator(
        pooling="mean", pooling_weights=weight_map_partial, default_pooling_weight=1
    )
    result = agg(sim_map)
    expected = (0.4 * 2 + 0.6 * 3 + 0.8 * 1) / (2 + 3 + 1)
    assert result == pytest.approx(expected)


def test_custom_pooling_function():
    """Test with custom pooling function."""

    def variance_pooling(values):
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    similarities = [0.2, 0.4, 0.6, 0.8, 1.0]
    agg = cbrkit.sim.aggregator(pooling=variance_pooling)

    expected = variance_pooling(similarities)
    assert agg(similarities) == pytest.approx(expected)


def test_edge_cases():
    """Test edge cases."""
    # Single value
    agg = cbrkit.sim.aggregator(pooling="mean")
    assert agg([0.5]) == pytest.approx(0.5)
    assert agg({"a": 0.5}) == pytest.approx(0.5)

    # Empty pooling weights
    agg = cbrkit.sim.aggregator(pooling="mean", pooling_weights=[])
    with pytest.raises(ZeroDivisionError):
        agg([])


def test_type_errors():
    """Test type mismatch and other errors."""
    # Type mismatch between similarities and weights
    agg = cbrkit.sim.aggregator(pooling="mean", pooling_weights=[1, 2])
    with pytest.raises(AssertionError):
        agg({"a": 0.5, "b": 0.6})

    # Strict zip error
    agg = cbrkit.sim.aggregator(pooling="mean", pooling_weights=[1, 2])
    with pytest.raises(ValueError):
        agg([0.5, 0.6, 0.7])


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
    agg = cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_min(k=k))
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
    agg = cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_max(k=k))
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
    agg = cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.minkowski(p=p))
    assert agg(similarities) == pytest.approx(formula(similarities))


def test_euclidean_pooling():
    """Test Euclidean pooling function."""
    similarities = [0.3, 0.4, 0.5]

    # Without weights
    agg = cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.euclidean())
    expected = math.sqrt(sum(s**2 for s in similarities))
    assert agg(similarities) == pytest.approx(expected)

    # Verify it's the same as minkowski with p=2
    agg_minkowski = cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.minkowski(p=2))
    assert agg(similarities) == pytest.approx(agg_minkowski(similarities))


def test_custom_pooling_edge_cases():
    """Test edge cases for custom pooling functions."""
    # k=0 should behave like k=1
    similarities = [0.1, 0.5, 0.9]
    assert cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_min(k=0))(
        similarities
    ) == pytest.approx(0.1)
    assert cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_max(k=0))(
        similarities
    ) == pytest.approx(0.9)

    # Single element
    assert cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_min(k=1))(
        [0.5]
    ) == pytest.approx(0.5)
    assert cbrkit.sim.aggregator(pooling=cbrkit.sim.pooling.k_max(k=1))(
        [0.5]
    ) == pytest.approx(0.5)
