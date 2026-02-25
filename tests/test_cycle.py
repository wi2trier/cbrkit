"""Integration tests for cbrkit pipelines and CBR cycles."""

import cbrkit
from cbrkit.helpers import unpack_float


def test_simple(cars_csv_casebase, car_query):
    casebase = cars_csv_casebase

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "manufacturer": cbrkit.sim.taxonomy.build(
                        "./data/cars-taxonomy.yaml",
                        cbrkit.sim.taxonomy.wu_palmer(),
                    ),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": cbrkit.sim.numbers.linear(max=100000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            )
        ),
        limit=5,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
                "make": cbrkit.adapt.strings.regex("a[0-9]", "a[0-9]", "a4"),
                "manufacturer": cbrkit.adapt.strings.regex(
                    "audi", "audi", "mercedes-benz"
                ),
                "miles": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "manufacturer": cbrkit.sim.taxonomy.build(
                    "./data/cars-taxonomy.yaml",
                    cbrkit.sim.taxonomy.wu_palmer(),
                ),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numbers.linear(max=100000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    )

    result = cbrkit.cycle.apply_queries(
        casebase, {"default": car_query}, retriever, reuser, [], []
    )

    assert len(result.retrieval.casebase) == len(result.reuse.casebase)


def test_multi_query_retrieval(cars_csv_casebase):
    """Test apply_queries with multiple queries simultaneously."""
    casebase = cars_csv_casebase
    query_keys = [10, 42]
    queries = {key: casebase[key] for key in query_keys}

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=5,
    )

    result = cbrkit.retrieval.apply_queries(casebase, queries, retriever)

    # Both query results should exist
    assert set(result.queries.keys()) == set(query_keys)

    # Each query should rank itself first (perfect self-similarity)
    for key in query_keys:
        query_result = result.queries[key]
        assert len(query_result.casebase) == 5
        assert query_result.ranking[0] == key
        assert unpack_float(query_result.similarities[key]) == 1.0


def test_multi_step_retrieval(cars_csv_casebase, car_query):
    """Test chaining two retrievers (MAC/FAC pattern)."""
    casebase = cars_csv_casebase

    # Coarse retriever: fast filter to top 50
    coarse_retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=50,
    )

    # Fine retriever: detailed comparison on top 5
    fine_retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=5,
    )

    result = cbrkit.retrieval.apply_query(
        casebase, car_query, [coarse_retriever, fine_retriever]
    )

    assert len(result.steps) == 2
    assert len(result.first_step.casebase) == 50
    assert len(result.final_step.casebase) == 5


def test_reuse_apply_result(cars_csv_casebase, car_query):
    """Test cbrkit.reuse.apply_result on a retrieval result."""
    casebase = cars_csv_casebase

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=5,
    )

    retrieval_result = cbrkit.retrieval.apply_query(casebase, car_query, retriever)
    assert len(retrieval_result.casebase) == 5

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
                "miles": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "miles": cbrkit.sim.numbers.linear(max=1000000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    )

    reuse_result = cbrkit.reuse.apply_result(retrieval_result, reuser)

    assert len(reuse_result.casebase) == 5

    # Reuse result should have valid similarities for all cases
    for score in reuse_result.similarities.values():
        assert 0.0 <= unpack_float(score) <= 1.0


def test_full_cycle_all_phases(medium_casebase, simple_query, sim_func_simple):
    """Test full 4-phase CBR cycle where all phases do meaningful work."""
    casebase = dict(medium_casebase)

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(sim_func_simple),
        limit=2,
    )

    reuser = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=sim_func_simple,
    )

    reviser = cbrkit.revise.build(assess_func=sim_func_simple)

    storage_func = cbrkit.retain.static(
        key_func=lambda keys: max(keys, default=-1) + 1,
        casebase=casebase,
    )
    retainer = cbrkit.retain.build(
        assess_func=sim_func_simple,
        storage_func=storage_func,
    )

    result = cbrkit.cycle.apply_query(
        casebase,
        simple_query,
        retriever,
        reuser,
        reviser,
        retainer,
    )

    # All four phases should have produced non-empty casebases
    assert len(result.retrieval.casebase) > 0
    assert len(result.reuse.casebase) > 0
    assert len(result.revise.casebase) > 0
    assert len(result.retain.casebase) > 0
    assert result.duration > 0


def test_eval_retrieval_integration(cars_csv_casebase):
    """Test cbrkit.eval on real retrieval results."""
    casebase = cars_csv_casebase
    query_keys = [10, 42]
    queries = {key: casebase[key] for key in query_keys}

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=10,
    )

    result = cbrkit.retrieval.apply_queries(casebase, queries, retriever)

    # Generate qrels from the result itself (self-evaluation)
    qrels = cbrkit.eval.retrieval_to_qrels(result, max_qrel=5)
    assert len(qrels) == 1  # one step

    # Compute metrics using the qrels
    metrics = cbrkit.eval.retrieval(
        qrels[0],
        result,
        metrics=["correctness", "completeness"],
    )

    assert len(metrics) == 1  # one step
    step_metrics = metrics[0]
    assert "correctness" in step_metrics
    assert "completeness" in step_metrics
    assert isinstance(step_metrics["correctness"], float)
    assert isinstance(step_metrics["completeness"], float)


def test_retrieve_combine_weighted(cars_csv_casebase, car_query):
    """Test cbrkit.retrieval.combine with two different similarity functions."""
    casebase = cars_csv_casebase

    retriever_price = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                },
            ),
        ),
        limit=20,
    )

    retriever_year = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "year": cbrkit.sim.numbers.linear(max=50),
                },
            ),
        ),
        limit=20,
    )

    combined = cbrkit.retrieval.combine(
        {"price": retriever_price, "year": retriever_year},
        aggregator=cbrkit.sim.aggregator(
            pooling="mean",
            pooling_weights={"price": 2, "year": 1},
        ),
    )

    result = cbrkit.retrieval.apply_query(casebase, car_query, combined)

    # Union strategy: should have cases from both retrievers
    assert len(result.casebase) > 0
    # All combined scores should be valid floats
    for score in result.similarities.values():
        assert 0.0 <= unpack_float(score) <= 1.0
