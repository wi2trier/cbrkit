from typing import Any

import cbrkit
from cbrkit.helpers import unpack_float


def test_revise_assess_only():
    """Test revision with assessment only (no repair)."""
    casebase = {
        0: {"price": 12000, "year": 2008, "miles": 150000},
        1: {"price": 9000, "year": 2012, "miles": 80000},
    }
    query = {"price": 10000, "year": 2010, "miles": 100000}

    reviser = cbrkit.revise.build(
        assess_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "miles": cbrkit.sim.numbers.linear(max=1000000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    )

    result = cbrkit.revise.apply_query(casebase, query, reviser)

    assert len(result.casebase) == 2
    assert len(result.similarities) == 2
    # All quality scores should be between 0 and 1
    for score in result.similarities.values():
        assert 0.0 <= unpack_float(score) <= 1.0


def test_revise_with_repair(small_casebase, simple_query, sim_func_simple):
    """Test revision with both repair and assessment."""
    casebase = dict(small_casebase)

    def repair_func(case: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
        # Simple repair: average price between case and query
        return {
            "price": (case["price"] + query["price"]) // 2,
            "year": case["year"],
        }

    reviser = cbrkit.revise.build(
        assess_func=sim_func_simple,
        repair_func=repair_func,
    )

    result = cbrkit.revise.apply_query(casebase, simple_query, reviser)

    assert len(result.casebase) == 2
    # Repaired prices should be averages
    assert result.casebase[0]["price"] == 11000
    assert result.casebase[1]["price"] == 9500


def test_revise_pair():
    """Test revision on a single case-query pair."""
    case = {"price": 12000}
    query = {"price": 10000}

    reviser = cbrkit.revise.build(
        assess_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
            },
        ),
    )

    result = cbrkit.revise.apply_pair(case, query, reviser)

    assert result.similarity is not None


def test_revise_result(medium_casebase, simple_query, sim_func_simple):
    """Test applying revision to a retrieval result."""
    casebase = dict(medium_casebase)

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(sim_func_simple),
        limit=2,
    )

    retrieval_result = cbrkit.retrieval.apply_query(casebase, simple_query, retriever)
    assert len(retrieval_result.casebase) == 2

    reviser = cbrkit.revise.build(assess_func=sim_func_simple)

    revise_result = cbrkit.revise.apply_result(retrieval_result, reviser)

    assert len(revise_result.casebase) == 2
    for score in revise_result.similarities.values():
        assert 0.0 <= unpack_float(score) <= 1.0
