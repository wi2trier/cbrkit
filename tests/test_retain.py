import cbrkit
from cbrkit.helpers import unpack_float

SIM_FUNC = cbrkit.sim.attribute_value(
    attributes={
        "price": cbrkit.sim.numbers.linear(max=100000),
        "year": cbrkit.sim.numbers.linear(max=50),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)

STORAGE_FUNC = cbrkit.retain.auto_key(
    key_func=lambda cb: max(cb.keys(), default=-1) + 1,
)


def test_retain_build():
    """Test retainer stores the query and produces real scores."""
    retainer = cbrkit.retain.build(
        assess_func=SIM_FUNC,
        storage_func=STORAGE_FUNC,
    )

    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    results = retainer([(casebase, query)])

    assert len(results) == 1
    updated_casebase, sim_map = results[0]
    assert len(updated_casebase) == 3
    assert updated_casebase[2] == query
    assert all(0.0 <= unpack_float(v) <= 1.0 for v in sim_map.values())
    assert unpack_float(sim_map[2]) == 1.0


def test_retain_dropout():
    """Test dropout keeps cases above min and rejects above max."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    inner = cbrkit.retain.build(assess_func=SIM_FUNC, storage_func=STORAGE_FUNC)

    # min_similarity=0.5: self-similarity is 1.0, so the case is kept
    kept = cbrkit.retain.dropout(inner, min_similarity=0.5)([(casebase, query)])
    assert len(kept[0][0]) == 3

    # min_similarity=1.1: impossible threshold, case is rejected
    rejected = cbrkit.retain.dropout(inner, min_similarity=1.1)([(casebase, query)])
    assert len(rejected[0][0]) == 2
    assert 2 not in rejected[0][0]

    # max_similarity=0.9: self-similarity 1.0 > 0.9, case is rejected
    novelty = cbrkit.retain.dropout(inner, max_similarity=0.9)([(casebase, query)])
    assert len(novelty[0][0]) == 2


def test_retain_full_cycle():
    """Test retain phase within a full CBR cycle."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
        2: {"price": 11000, "year": 2010},
    }
    query = {"price": 10000, "year": 2010}

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(SIM_FUNC),
        limit=2,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=SIM_FUNC,
    )

    reviser = cbrkit.revise.build(assess_func=SIM_FUNC)

    retainer = cbrkit.retain.build(
        assess_func=SIM_FUNC,
        storage_func=STORAGE_FUNC,
    )

    result = cbrkit.cycle.apply_query(
        casebase, query, retriever, reuser, reviser, retainer,
    )

    assert len(result.retrieval.casebase) == 2
    assert len(result.retain.casebase) > 0
