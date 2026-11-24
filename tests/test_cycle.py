"""Integration tests for cbrkit pipelines and CBR cycles."""

import asyncio

import cbrkit


def test_simple(cars_csv_casebase, car_query):
    casebase = cars_csv_casebase

    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                    taxonomy=cbrkit.sim.taxonomy.load("./data/cars-taxonomy.yaml"),
                    measure=cbrkit.sim.taxonomy.WuPalmer(),
                ),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": cbrkit.sim.numbers.Linear(max_distance=100000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )

    reuser = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "price": cbrkit.adapt.numbers.Aggregate(),
                "make": cbrkit.adapt.strings.RegexAdaptation(
                    case_pattern="a[0-9]", query_pattern="a[0-9]", replacement="a4"
                ),
                "manufacturer": cbrkit.adapt.strings.RegexAdaptation(
                    case_pattern="audi",
                    query_pattern="audi",
                    replacement="mercedes-benz",
                ),
                "miles": cbrkit.adapt.numbers.Aggregate(),
            }
        ),
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                    taxonomy=cbrkit.sim.taxonomy.load("./data/cars-taxonomy.yaml"),
                    measure=cbrkit.sim.taxonomy.WuPalmer(),
                ),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": cbrkit.sim.numbers.Linear(max_distance=100000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
    )

    result = cbrkit.cycle.run(casebase, {"q": car_query}, retriever, reuser)
    retrieval, reuse = (step["q"] for step in result.steps)

    assert len(retrieval.casebase) == len(reuse.casebase) == 5
    assert result["q"] == reuse


def test_multi_query_retrieval(cars_csv_casebase):
    """Test run with multiple queries simultaneously."""
    casebase = cars_csv_casebase
    query_keys = [10, 42]
    queries = {key: casebase[key] for key in query_keys}

    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )

    result = cbrkit.cycle.run(casebase, queries, retriever)

    # Both query results should exist
    assert set(result) == set(query_keys)

    # Each query should rank itself first (perfect self-similarity)
    for key in query_keys:
        query_result = result[key]
        assert len(query_result.casebase) == 5
        assert query_result.ranking[0] == key
        assert float(query_result.similarities[key]) == 1.0


def test_multi_step_retrieval(cars_csv_casebase, car_query):
    """Test chaining two retrievers (MAC/FAC pattern)."""
    casebase = cars_csv_casebase

    # Coarse retriever: fast filter to top 50
    coarse_retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=50,
    )

    # Fine retriever: detailed comparison on top 5
    fine_retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )

    result = cbrkit.cycle.run(
        casebase, {"q": car_query}, coarse_retriever, fine_retriever
    )

    assert len(result.steps) == 2
    assert len(result.steps[0]["q"].casebase) == 50
    assert len(result["q"].casebase) == 5


def test_reuse_retrieved_cases(cars_csv_casebase, car_query):
    """A reuser applies to the cases of an earlier retrieval."""
    casebase = cars_csv_casebase

    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )

    retrieval_result = cbrkit.cycle.run_query(casebase, car_query, retriever)
    assert len(retrieval_result.casebase) == 5

    reuser = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "price": cbrkit.adapt.numbers.Aggregate(),
                "miles": cbrkit.adapt.numbers.Aggregate(),
            }
        ),
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
    )

    reuse_result = cbrkit.cycle.run_query(
        retrieval_result.casebase, retrieval_result.query, reuser
    )

    assert len(reuse_result.casebase) == 5

    # Reuse result should have valid similarities for all cases
    for score in reuse_result.similarities.values():
        assert 0.0 <= float(score) <= 1.0


def test_full_cycle_all_phases(medium_casebase, simple_query, sim_func_simple):
    """Test full 4-phase CBR cycle where all phases do meaningful work."""
    casebase = dict(medium_casebase)

    retriever = cbrkit.retrieve.Retriever(measure=sim_func_simple, limit=2)

    reuser = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "price": cbrkit.adapt.numbers.Aggregate(),
            }
        ),
        measure=sim_func_simple,
    )

    reviser = cbrkit.revise.Reviser(measure=sim_func_simple)

    storage = cbrkit.indexable.InMemory(casebase)
    retainer = cbrkit.retain.Retainer(storage=storage, measure=sim_func_simple)

    result = cbrkit.cycle.run(
        storage,
        {"q": simple_query},
        retriever,
        reuser,
        reviser,
        retainer,
    )

    # The four phases of the cycle produced non-empty casebases, and retain stored the solutions
    assert len(result.steps) == 4
    assert all(step["q"].casebase for step in result.steps)
    assert result.duration > 0
    assert [change.op for change in result.changes.items] == ["add", "add"]
    assert len(asyncio.run(storage.get_index())) == len(casebase) + 2


def test_eval_retrieval_integration(cars_csv_casebase):
    """Test cbrkit.eval on real retrieval results."""
    casebase = cars_csv_casebase
    query_keys = [10, 42]
    queries = {key: casebase[key] for key in query_keys}

    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=10,
    )

    result = cbrkit.cycle.run(casebase, queries, retriever)

    # Generate qrels from the result itself (self-evaluation)
    qrels = cbrkit.eval.to_qrels(result, max_qrel=5)
    metrics = cbrkit.eval.evaluate(
        qrels,
        result,
        metrics=[
            cbrkit.eval.Metric(name="correctness"),
            cbrkit.eval.Metric(name="completeness"),
        ],
    )

    assert isinstance(metrics[cbrkit.eval.Metric(name="correctness")], float)
    assert isinstance(metrics[cbrkit.eval.Metric(name="completeness")], float)


def test_retrieve_combine_weighted(cars_csv_casebase, car_query):
    """Test cbrkit.retrieve.combine with two different similarity functions."""
    casebase = cars_csv_casebase

    retriever_price = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
            },
        ),
        limit=20,
    )

    retriever_year = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
            },
        ),
        limit=20,
    )

    combined = cbrkit.retrieve.Combine(
        retrievers=[retriever_price, retriever_year],
        aggregator=cbrkit.sim.Aggregator(weights={0: 2, 1: 1}),
    )

    result = cbrkit.cycle.run_query(casebase, car_query, combined)

    # A default similarity keeps the cases from both retrievers
    assert len(result.casebase) > 0
    # All combined scores should be valid floats
    for score in result.similarities.values():
        assert 0.0 <= float(score) <= 1.0
