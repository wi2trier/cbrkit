import cbrkit
from cbrkit.helpers import unpack_float


def test_retain_auto_key_storage():
    """Test auto-key storage function."""
    storage = cbrkit.retain.auto_key(
        key_func=lambda cb: max(cb.keys(), default=-1) + 1,
    )

    casebase: dict[int, str] = {0: "a", 1: "b"}
    updated = storage(casebase, "c")

    assert 2 in updated
    assert updated[2] == "c"
    assert len(updated) == 3


def test_retain_build():
    """Test building a retainer from storage function."""
    retainer = cbrkit.retain.build(
        storage_func=cbrkit.retain.auto_key(
            key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ),
    )

    results = retainer(
        [({0: "case_a", 1: "case_b"}, "query")],
    )

    assert len(results) == 1
    updated_casebase, sim_map = results[0]
    assert len(updated_casebase) == 3
    assert updated_casebase[2] == "query"
    assert len(sim_map) == 3


def test_retain_apply_query():
    """Test applying retainer to a single query."""
    retainer = cbrkit.retain.build(
        storage_func=cbrkit.retain.auto_key(
            key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ),
    )

    casebase = {0: "case_a", 1: "case_b"}

    result = cbrkit.retain.apply_query(casebase, "query", retainer)

    assert len(result.casebase) == 3
    assert result.casebase[2] == "query"
    assert len(result.similarities) == 3


def test_retain_apply_result():
    """Test applying retainer to a result from revise/reuse."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    retriever = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        )
    )

    retrieval_result = cbrkit.retrieval.apply_query(casebase, query, retriever)

    retainer = cbrkit.retain.build(
        storage_func=cbrkit.retain.auto_key(
            key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ),
    )

    retain_result = cbrkit.retain.apply_result(retrieval_result, retainer)

    assert len(retain_result.casebase) == 3
    assert len(retain_result.similarities) == 3


def test_full_cycle_with_revise_and_retain():
    """Test full CBR cycle with all 4 phases."""
    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
        2: {"price": 11000, "year": 2010},
    }
    query = {"price": 10000, "year": 2010}

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            )
        ),
        limit=2,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        retriever_func=cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            )
        ),
    )

    reviser = cbrkit.revise.build(
        assess_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    )

    retainer = cbrkit.retain.build(
        storage_func=cbrkit.retain.auto_key(
            key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ),
    )

    result = cbrkit.cycle.apply_query(
        casebase,
        query,
        retriever,
        reuser,
        reviser,
        retainer,
    )

    assert result.retrieval is not None
    assert result.reuse is not None
    assert result.revise is not None
    assert result.retain is not None
    assert len(result.retrieval.casebase) == 2


def test_retain_build_with_assess_func():
    """Test retainer with an assessment function producing real scores."""
    sim_func = cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retainer = cbrkit.retain.build(
        storage_func=cbrkit.retain.auto_key(
            key_func=lambda cb: max(cb.keys(), default=-1) + 1,
        ),
        assess_func=sim_func,
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
    # Scores should be real similarity values, not all 1.0
    assert all(0.0 <= unpack_float(v) <= 1.0 for v in sim_map.values())
    # The newly added case (query itself) should have perfect self-similarity
    assert unpack_float(sim_map[2]) == 1.0


def test_retain_dropout_keeps_good_cases():
    """Test dropout keeps cases above the threshold."""
    sim_func = cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retainer = cbrkit.retain.dropout(
        retainer_func=cbrkit.retain.build(
            storage_func=cbrkit.retain.auto_key(
                key_func=lambda cb: max(cb.keys(), default=-1) + 1,
            ),
            assess_func=sim_func,
        ),
        min_similarity=0.5,
    )

    casebase = {
        0: {"price": 10000, "year": 2010},
        1: {"price": 10500, "year": 2010},
    }
    # Query is very similar to existing cases, self-similarity is 1.0
    query = {"price": 10000, "year": 2010}

    results = retainer([(casebase, query)])

    assert len(results) == 1
    updated_casebase, sim_map = results[0]
    # Case should be kept (self-similarity = 1.0 >= 0.5)
    assert len(updated_casebase) == 3
    assert 2 in updated_casebase


def test_retain_dropout_rejects_bad_cases():
    """Test dropout rejects cases below the threshold."""
    sim_func = cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retainer = cbrkit.retain.dropout(
        retainer_func=cbrkit.retain.build(
            storage_func=cbrkit.retain.auto_key(
                key_func=lambda cb: max(cb.keys(), default=-1) + 1,
            ),
            assess_func=sim_func,
        ),
        # Set threshold impossibly high so the new case is rejected
        min_similarity=1.1,
    )

    casebase = {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }
    query = {"price": 10000, "year": 2010}

    results = retainer([(casebase, query)])

    assert len(results) == 1
    updated_casebase, sim_map = results[0]
    # Case should be rejected (no score can exceed 1.1)
    assert len(updated_casebase) == 2
    assert 2 not in updated_casebase
    assert 2 not in sim_map


def test_retain_dropout_max_similarity():
    """Test dropout rejects cases above max_similarity (novelty gate)."""
    sim_func = cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retainer = cbrkit.retain.dropout(
        retainer_func=cbrkit.retain.build(
            storage_func=cbrkit.retain.auto_key(
                key_func=lambda cb: max(cb.keys(), default=-1) + 1,
            ),
            assess_func=sim_func,
        ),
        # Reject cases that are too similar (self-similarity = 1.0)
        max_similarity=0.9,
    )

    casebase = {
        0: {"price": 12000, "year": 2008},
    }
    query = {"price": 10000, "year": 2010}

    results = retainer([(casebase, query)])

    assert len(results) == 1
    updated_casebase, sim_map = results[0]
    # New case has self-similarity 1.0 > 0.9, so it should be rejected
    assert len(updated_casebase) == 1
    assert 1 not in updated_casebase
