import random
from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import pytest

from cbrkit.sim.collections import TWED
from cbrkit.sim.generic import Equality


def abs_diff_sim(x: float, y: float) -> float:
    return 1.0 - abs(x - y)


def value_sim(x: tuple[float, float], y: tuple[float, float]) -> float:
    return 1.0 - abs(x[1] - y[1])


def timestamp(value: tuple[float, float]) -> float:
    return value[0]


equality = Equality()


def constant(value: float) -> Callable[[Any, Any], float]:
    return lambda x, y: value


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
    sim = TWED(stiffness=stiffness, penalty=penalty, **kwargs)(x, y)

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
    assert distance(A, B, element_measure=abs_diff_sim) == pytest.approx(distance(A, B))


def test_timestamps_are_used() -> None:
    """Irregular timestamps must change the result once the stiffness is non-zero."""
    values = [(0, 1.0), (1, 2.0), (2, 3.0)]
    stretched = [(0, 1.0), (10, 2.0), (20, 3.0)]
    sim = TWED(
        element_measure=value_sim,
        timestamp=timestamp,
        stiffness=0.1,
    )

    assert sim(values, values).value == 1.0
    assert sim(values, stretched).value < 1.0


@pytest.mark.parametrize(
    "invalid",
    [
        [(10.0, 0.0), (0.0, 0.0)],
        [(float("nan"), 0.0)],
        [(float("inf"), 0.0)],
    ],
)
def test_invalid_timestamps(invalid: list[tuple[float, float]]) -> None:
    sim = TWED(element_measure=value_sim, timestamp=timestamp, stiffness=1.0)

    with pytest.raises(ValueError, match="finite and nondecreasing"):
        sim(invalid, [(0.0, 0.0)])

    with pytest.raises(ValueError, match="finite and nondecreasing"):
        sim([(0.0, 0.0)], invalid)


def test_alignment_covers_both_sequences() -> None:
    result = TWED(alignment=True)(A, B)

    assert result.mapping is not None
    assert result.similarities is not None
    assert len(result.mapping) == len(result.similarities)
    assert [case for _, case in result.mapping if case is not None] == A
    assert [query for query, _ in result.mapping if query is not None] == B


def test_multivariate_series() -> None:
    x = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    assert TWED()(x, x).value == 1.0
    assert TWED()(x, [[1.0, 1.0], [9.0, 9.0], [3.0, 3.0]]).value < 1.0


def test_empty_sequences() -> None:
    assert TWED()([], []).value == 1.0
    assert TWED()([1, 2], []).value == 0.0


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        TWED(stiffness=-1.0)

    with pytest.raises(ValueError):
        TWED(penalty=-1.0)


def test_normalized_similarity_scale() -> None:
    """Identical sequences score 1, maximally different ones 0, and a single
    substitution is penalized less in a longer sequence."""
    sim = TWED(element_measure=equality, normalize=True)

    assert sim(list("abc"), list("abc")).value == 1.0
    assert sim(list("abc"), list("xyz")).value == pytest.approx(0.0)
    assert sim(list("ab"), list("wxyz")).value == pytest.approx(0.0)
    assert sim(list("abcde"), list("abxde")).value == pytest.approx(7 / 9)
    assert sim(list("abcdefghij"), list("abcdexghij")).value == pytest.approx(17 / 19)


def test_normalized_singletons_return_the_local_similarity() -> None:
    sim = TWED(element_measure=constant(0.7), normalize=True)

    assert sim(["a"], ["b"]).value == pytest.approx(0.7)


def test_normalized_uniform_distance_is_length_independent() -> None:
    sim = TWED(element_measure=constant(0.9), normalize=True)

    assert sim(list("abc"), list("def")).value == pytest.approx(0.9)
    assert sim(list("abcdefgh"), list("ijklmnop")).value == pytest.approx(0.9)


@pytest.mark.parametrize(("n", "m"), [(1, 1), (3, 3), (2, 5), (6, 1)])
def test_max_distance_closed_form(n: int, m: int) -> None:
    """With the default timestamps, the bound has a closed form."""
    stiffness, penalty = 0.01, 0.5
    sim = TWED(stiffness=stiffness, penalty=penalty)

    assert sim.max_distance(range(n), range(m)) == pytest.approx(
        2 * min(n, m) - 1 + abs(n - m) * (1 + stiffness + penalty)
    )


def test_max_distance_bounds_the_distance() -> None:
    """Sample distances in [0, 1] never exceed the bound, also with irregular
    timestamps, and the bound is reached if all sample distances are 1."""
    rng = random.Random(42)

    for _ in range(200):
        x = [(rng.uniform(0, 10), rng.random()) for _ in range(rng.randint(1, 7))]
        y = [(rng.uniform(0, 10), rng.random()) for _ in range(rng.randint(1, 7))]
        x.sort()
        y.sort()
        sim = TWED(
            element_measure=value_sim,
            timestamp=timestamp,
            stiffness=rng.choice([0.0, 0.01, 1.0]),
            penalty=rng.choice([0.0, 0.5, 1.0]),
            normalize=True,
        )
        bound = sim.max_distance(x, y)
        distance, _, _ = sim.compute(x, y)

        assert distance <= bound + 1e-9
        assert 0.0 <= sim(x, y).value <= 1.0

        worst = TWED(
            element_measure=constant(0.0),
            timestamp=timestamp,
            stiffness=sim.stiffness,
            penalty=sim.penalty,
            normalize=True,
        )
        assert worst(x, y).value == pytest.approx(0.0)


def test_normalized_alignment_similarities() -> None:
    sim = TWED(element_measure=constant(0.75), normalize=True, alignment=True)
    result = sim(list("ab"), list("cd"))

    assert result.similarities == (0.75, 0.75)


def test_normalize_rejects_distances_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        TWED(normalize=True)([0, 5], [0, 1])


def test_normalized_empty_sequences() -> None:
    assert TWED(normalize=True)([], []).value == 1.0
    assert TWED(normalize=True)([1, 2], []).value == 0.0
