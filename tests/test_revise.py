from typing import Any

import pytest

import cbrkit


def test_revise_assess_only():
    """Test revision with assessment only (no repair)."""
    casebase = {
        0: {"price": 12000, "year": 2008, "miles": 150000},
        1: {"price": 9000, "year": 2012, "miles": 80000},
    }
    query = {"price": 10000, "year": 2010, "miles": 100000}

    reviser = cbrkit.revise.Reviser(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
    )

    result = cbrkit.cycle.run_query(casebase, query, reviser)

    assert len(result.casebase) == 2
    assert len(result.similarities) == 2
    # All quality scores should be between 0 and 1
    for score in result.similarities.values():
        assert 0.0 <= float(score) <= 1.0


def test_revise_with_repair(small_casebase, simple_query, sim_func_simple):
    """Failing solutions are repaired and assessed again, passing ones are kept as they are."""
    casebase = {**small_casebase, 2: dict(simple_query)}

    def repair(case: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        return {**case, "price": query["price"], "year": query["year"]}

    reviser = cbrkit.revise.Reviser(
        measure=sim_func_simple, repair=repair, min_similarity=0.99
    )
    result = cbrkit.cycle.run_query(casebase, simple_query, reviser)

    assert result.casebase == {0: simple_query, 1: simple_query, 2: simple_query}
    assert result.casebase[2] is casebase[2]
    assert all(float(value) == 1.0 for value in result.similarities.values())

    with pytest.raises(ValueError, match="min_similarity"):
        cbrkit.revise.Reviser(measure=sim_func_simple, repair=repair)


def test_revise_pair():
    """Test revision on a single case-query pair."""
    case = {"price": 12000}
    query = {"price": 10000}

    reviser = cbrkit.revise.Reviser(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
            },
        ),
    )

    result = cbrkit.cycle.run_query({"case": case}, query, reviser)

    assert float(result.similarities["case"]) == 0.98


def test_revise_result(medium_casebase, simple_query, sim_func_simple):
    """Test applying revision to a retrieval result."""
    casebase = dict(medium_casebase)

    retriever = cbrkit.retrieve.Retriever(measure=sim_func_simple, limit=2)

    retrieval_result = cbrkit.cycle.run_query(casebase, simple_query, retriever)
    assert len(retrieval_result.casebase) == 2

    reviser = cbrkit.revise.Reviser(measure=sim_func_simple)

    revise_result = cbrkit.cycle.run_query(
        retrieval_result.casebase, retrieval_result.query, reviser
    )

    assert len(revise_result.casebase) == 2
    for score in revise_result.similarities.values():
        assert 0.0 <= float(score) <= 1.0
