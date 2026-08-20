from collections.abc import Sequence
from itertools import product
from typing import Any

import pytest

from cbrkit.sim.collections import twed


def abs_diff(x: float, y: float) -> float:
    return abs(x - y)


def value_diff(x: tuple[float, float], y: tuple[float, float]) -> float:
    return abs(x[1] - y[1])


def timestamp(value: tuple[float, float]) -> float:
    return value[0]


# Reference series and parameters from the __main__ block of
# https://github.com/pfmarteau/TWED/blob/master/twed.py
A = [0, 0, 1, 1, 2, 3, 5, 2, 0, 1, -0.1]
B = [0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5.5, 2, 0, 0, 0.25, 0.05, 0]
C = [4, 4, 3, 3, 3, 3, 2, 5, 2, 0.5, 0.5, 0.5]


def distance(
    x: Sequence[Any],
    y: Sequence[Any],
    stiffness: float = 0.1,
    penalty: float = 0.2,
    **kwargs: Any,
) -> float:
    """Invert the similarity conversion to recover the raw TWED distance."""
    sim = twed(stiffness=stiffness, penalty=penalty, **kwargs)(x, y)

    return 1 / sim.value - 1


def test_reference_values() -> None:
    assert distance(A, B) == pytest.approx(11.9)
    assert distance(A, C) == pytest.approx(16.3)
    assert distance(B, C) == pytest.approx(19.9)


def test_metric_properties() -> None:
    for u in (A, B, C):
        assert distance(u, u) == 0.0

    for u, v in product((A, B, C), repeat=2):
        assert distance(u, v) == pytest.approx(distance(v, u))

    assert distance(A, C) <= distance(A, B) + distance(B, C) + 1e-9
    assert distance(A, B) <= distance(A, C) + distance(B, C) + 1e-9
    assert distance(B, C) <= distance(A, B) + distance(A, C) + 1e-9


def test_monotone_in_parameters() -> None:
    """Proposition 3 of the paper: the distance increases with stiffness and penalty."""
    stiffness_distances = [distance(A, B, stiffness=s) for s in (0.0, 0.01, 0.1, 1.0)]
    penalty_distances = [distance(A, B, penalty=p) for p in (0.0, 0.2, 1.0, 5.0)]

    assert stiffness_distances == sorted(stiffness_distances)
    assert penalty_distances == sorted(penalty_distances)


def test_custom_distance_matches_builtin() -> None:
    assert distance(A, B, distance_func=abs_diff) == pytest.approx(distance(A, B))


def test_timestamps_are_used() -> None:
    """Irregular timestamps must change the result once the stiffness is non-zero."""
    values = [(0, 1.0), (1, 2.0), (2, 3.0)]
    stretched = [(0, 1.0), (10, 2.0), (20, 3.0)]
    sim = twed(
        distance_func=value_diff,
        timestamp_func=timestamp,
        stiffness=0.1,
    )

    assert sim(values, values).value == 1.0
    assert sim(values, stretched).value < 1.0


def test_alignment_covers_both_sequences() -> None:
    result = twed()(A, B, return_alignment=True)

    assert result.mapping is not None
    assert result.similarities is not None
    assert len(result.mapping) == len(result.similarities)
    assert [case for _, case in result.mapping if case is not None] == A
    assert [query for query, _ in result.mapping if query is not None] == B


def test_multivariate_series() -> None:
    x = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    assert twed()(x, x).value == 1.0
    assert twed()(x, [[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]]).value < 1.0


def test_empty_sequences() -> None:
    assert twed()([], []).value == 1.0
    assert twed()([1, 2], []).value == 0.0


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        twed(stiffness=-1.0)

    with pytest.raises(ValueError):
        twed(penalty=-1.0)
