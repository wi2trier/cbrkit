"""Regression tests for the repaired sequence and graph alignment measures.

Every test here pins down a defect that was only observable at runtime, so the
docstring of each test names the behaviour that used to be wrong.
"""

import random
from collections.abc import Callable
from itertools import permutations
from typing import Any

import pytest

import cbrkit
from cbrkit.helpers import unbatchify
from cbrkit.model.graph import Graph, from_dict
from cbrkit.sim.collections import DTW, TWED, OptimalMapping, SmithWaterman
from cbrkit.sim.generic import Equality

type NodeSim = Callable[[Any, Any], float]


equality = Equality()


ALIGNERS: dict[str, Callable[..., Any]] = {
    "dtw": cbrkit.sim.graphs.GraphDTW,
    "twed": cbrkit.sim.graphs.GraphTWED,
    "smith_waterman": cbrkit.sim.graphs.GraphSmithWaterman,
}


def aligner(name: str) -> Callable[[Any, Any], Any]:
    return unbatchify(ALIGNERS[name](node_measure=equality))


def weighted_sim(weights: list[list[float]]) -> NodeSim:
    def sim(case: int, query: int) -> float:
        return weights[query][case]

    return sim


def graph(
    nodes: dict[str, str], edges: list[tuple[str, str, str]]
) -> Graph[Any, Any, Any, Any]:
    return from_dict(
        {
            "nodes": nodes,
            "edges": {
                key: {"source": source, "target": target, "value": None}
                for key, source, target in edges
            },
            "value": None,
        }
    )


CHAIN_X = graph({"1": "A", "2": "B", "3": "C"}, [("e1", "1", "2"), ("e2", "2", "3")])
CHAIN_Y = graph({"1": "A", "2": "X", "3": "C"}, [("e1", "1", "2"), ("e2", "2", "3")])
SHORT_Y = graph({"1": "A", "2": "B"}, [("e1", "1", "2")])


@pytest.mark.parametrize("name", ALIGNERS.keys())
def test_self_similarity_is_one(name: str) -> None:
    """D1/D2: the aligners used to score a graph against itself at 0.22 and 0.02."""
    assert aligner(name)(CHAIN_X, CHAIN_X).value == pytest.approx(1.0)


@pytest.mark.parametrize("name", ALIGNERS.keys())
def test_agrees_with_astar(name: str) -> None:
    """D2: the aligners normalized twice, so their scale was not comparable to A*."""
    baseline = unbatchify(cbrkit.sim.graphs.Astar(node_measure=equality))

    for query in (CHAIN_X, CHAIN_Y, SHORT_Y):
        expected = baseline(CHAIN_X, query).value
        assert aligner(name)(CHAIN_X, query).value == pytest.approx(expected)


@pytest.mark.parametrize("name", ALIGNERS.keys())
def test_mapping_is_injective_and_query_keyed(name: str) -> None:
    """D1/D7/D8: warping produced duplicate targets, and dtw keyed the mapping by case."""
    case = graph({"c1": "A", "c2": "B"}, [("ce", "c1", "c2")])
    query = graph(
        {"q1": "A", "q2": "A", "q3": "B"}, [("qe1", "q1", "q2"), ("qe2", "q2", "q3")]
    )
    result = aligner(name)(case, query)

    assert set(result.mapping.keys()) <= set(query.nodes.keys())
    assert set(result.mapping.values()) <= set(case.nodes.keys())
    assert len(set(result.mapping.values())) == len(result.mapping)


@pytest.mark.parametrize("name", ALIGNERS.keys())
def test_differing_edge_counts(name: str) -> None:
    """D3: a strict zip over the edge lists raised ValueError for differing counts."""
    result = aligner(name)(CHAIN_X, SHORT_Y)

    assert 0.0 <= result.value <= 1.0


def test_smith_waterman_uses_node_values() -> None:
    """D4: the alignment was computed over position indices, so only lengths mattered."""
    sim = aligner("smith_waterman")
    same_shape_worse = graph(
        {"1": "X", "2": "Y", "3": "Z"}, [("e1", "1", "2"), ("e2", "2", "3")]
    )

    assert sim(CHAIN_X, CHAIN_X).value > sim(CHAIN_X, CHAIN_Y).value
    assert sim(CHAIN_X, CHAIN_Y).value > sim(CHAIN_X, same_shape_worse).value


@pytest.mark.parametrize("name", ALIGNERS.keys())
def test_retrieval_pipeline(name: str) -> None:
    """The aligners must rank the identical graph first when used for retrieval.

    The graphs in `data/graphs-v1.json` are cyclic, so they cannot be turned into a
    sequence and a small sequential casebase is used instead.
    """
    casebase = {"exact": CHAIN_X, "partial": CHAIN_Y, "short": SHORT_Y}
    retriever = cbrkit.retrieve.Retriever(measure=ALIGNERS[name](node_measure=equality))
    result = cbrkit.cycle.run_query(casebase, CHAIN_X, retriever)

    assert result.similarities["exact"].value == pytest.approx(1.0)
    assert result.ranking[0] == "exact"


def test_smith_waterman_sequence_is_normalized() -> None:
    """D5: the sequence measure returned an unnormalized int such as 2 or 20."""
    sim = SmithWaterman()

    assert sim("abcde", "abcde").value == 1.0
    assert sim("abcdefghij", "abcdefghij").value == 1.0

    for x in ("", "a", "abcde", "abcdefghij"):
        for y in ("", "z", "abc", "abcdefghij"):
            assert 0.0 <= sim(x, y).value <= 1.0


def test_dtw_prefers_the_diagonal() -> None:
    """D8: ties were broken towards a gap, so identical sequences produced 3 steps."""
    result = DTW(alignment=True)([0, 0], [0, 0])

    assert result.mapping == ((0, 0), (0, 0))


def test_dtw_calls_the_element_measure_once_per_pair() -> None:
    """D9: the backtracking used to re-invoke the distance function."""
    calls: list[tuple[float, float]] = []

    def element_measure(a: float, b: float) -> float:
        calls.append((a, b))

        return 1.0 - abs(a - b)

    DTW(element_measure=element_measure, alignment=True)([1, 2, 3], [1, 2, 3, 4])

    assert len(calls) == 12


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

        assert float(
            OptimalMapping(element_measure=weighted_sim(weights))(items, items)
        ) == pytest.approx(expected / size)


def test_mapping_edge_cases() -> None:
    """D6: duplicates collapsed, a longer query returned 0.0, and empty input raised."""
    sim = OptimalMapping(element_measure=equality)

    assert float(sim(["a", "a"], ["a", "a"])) == 1.0
    assert float(sim(["a", "b", "a"], ["a", "b", "a"])) == 1.0
    assert float(sim(["a"], ["a", "b", "c"])) == pytest.approx(1 / 3)
    assert float(sim([], [])) == 1.0
    assert float(sim([], ["a"])) == 0.0
    assert float(sim([[1]], [[1]])) == 1.0


def test_twed_graph_alignment_is_injective_without_filtering() -> None:
    """TWED deletes unmatched nodes, so its raw alignment is already injective."""
    x = [1, 2, 3]
    y = [1, 5, 3, 9]
    result = TWED(alignment=True)(x, y)

    assert result.mapping is not None

    cases = [case for _, case in result.mapping if case is not None]
    queries = [query for query, _ in result.mapping if query is not None]

    assert cases == x
    assert queries == y
