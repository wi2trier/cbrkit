"""Regression tests for the repaired sequence and graph alignment measures.

Every test here pins down a defect that was only observable at runtime, so the
docstring of each test names the behaviour that used to be wrong.
"""

import random
from collections.abc import Callable
from itertools import permutations
from typing import Any

import pytest

from cbrkit.sim.collections import dtw, mapping, smith_waterman, twed

type NodeSim = Callable[[Any, Any], float]


def equality(x: Any, y: Any) -> float:
    return 1.0 if x == y else 0.0


def abs_diff(x: float, y: float) -> float:
    return abs(x - y)


def weighted_sim(weights: list[list[float]]) -> NodeSim:
    def sim(case: int, query: int) -> float:
        return weights[query][case]

    return sim


def test_smith_waterman_sequence_is_normalized() -> None:
    """D5: the sequence measure returned an unnormalized int such as 2 or 20."""
    sim = smith_waterman()

    assert sim("abcde", "abcde").value == 1.0
    assert sim("abcdefghij", "abcdefghij").value == 1.0

    for x in ("", "a", "abcde", "abcdefghij"):
        for y in ("", "z", "abc", "abcdefghij"):
            assert 0.0 <= sim(x, y).value <= 1.0


def test_dtw_prefers_the_diagonal() -> None:
    """D8: ties were broken towards a gap, so identical sequences produced 3 steps."""
    result = dtw()([0, 0], [0, 0], return_alignment=True)

    assert result.mapping == [(0, 0), (0, 0)]


def test_dtw_calls_the_distance_func_once_per_pair() -> None:
    """D9: the backtracking used to re-invoke the distance function."""
    calls = 0

    def distance_func(a: float, b: float) -> float:
        nonlocal calls
        calls += 1

        return abs(a - b)

    dtw(distance_func)([1, 2, 3], [1, 2, 3, 4], return_alignment=True)

    assert calls == 12


def test_mapping_is_optimal() -> None:
    """D6: the search expanded the worst state first and lost the optimum from n=7."""
    rng = random.Random(0)

    for size in range(1, 7):
        weights = [[rng.random() for _ in range(size)] for _ in range(size)]
        items = list(range(size))
        expected = max(
            sum(weights[query][case] for query, case in enumerate(perm))
            for perm in permutations(items)
        )

        assert mapping(weighted_sim(weights))(items, items) == pytest.approx(
            expected / size
        )


def test_mapping_edge_cases() -> None:
    """D6: duplicates collapsed, a longer query returned 0.0, and empty input raised."""
    sim = mapping(equality)

    assert sim(["a", "a"], ["a", "a"]) == 1.0
    assert sim(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    assert sim(["a"], ["a", "b", "c"]) == pytest.approx(1 / 3)
    assert sim([], []) == 0.0
    assert sim([[1]], [[1]]) == 1.0


def test_twed_graph_alignment_is_injective_without_filtering() -> None:
    """TWED deletes unmatched nodes, so its raw alignment is already injective."""
    x = [1, 2, 3]
    y = [1, 5, 3, 9]
    result = twed(distance_func=abs_diff)(x, y, return_alignment=True)

    assert result.mapping is not None

    cases = [case for _, case in result.mapping if case is not None]
    queries = [query for query, _ in result.mapping if query is not None]

    assert cases == x
    assert queries == y
