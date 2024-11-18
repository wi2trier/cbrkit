from typing import Any

import polars as pl

import cbrkit


def _custom_numeric_sim(x: float, y: float) -> float:
    return 1 - abs(x - y) / 100000


def test_retrieve_multiprocessing():
    query_name = 42
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)
    retriever = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "manufacturer": cbrkit.sim.strings.taxonomy.load(
                    "./data/cars-taxonomy.yaml",
                    measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
                ),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": _custom_numeric_sim,
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        limit=5,
    )
    results = cbrkit.retrieval.mapply(
        casebase,
        {"default": casebase[query_name]},
        retriever,
        processes=2,
    )

    assert len(results) == 1
    assert len(results["default"].ranking) == 5

    result = cbrkit.retrieval.apply(
        casebase,
        casebase[query_name],
        retriever,
        processes=2,
    )

    assert len(result.ranking) == 5


def test_retrieve_dataframe():
    query_name = 42
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "manufacturer": cbrkit.sim.strings.taxonomy.load(
                    "./data/cars-taxonomy.yaml",
                    measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
                ),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numbers.linear(max=1000000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        limit=5,
    )
    result = cbrkit.retrieval.apply(casebase, query, retriever)

    assert len(casebase) == 999  # csv contains header
    assert len(result.similarities) == 5
    assert len(result.ranking) == 5
    assert len(result.casebase) == 5
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name


def test_retrieve_dataframe_custom_query():
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)

    query = {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }

    retriever = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "manufacturer": cbrkit.sim.strings.taxonomy.load(
                    "./data/cars-taxonomy.yaml",
                    measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
                ),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": _custom_numeric_sim,
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        limit=5,
    )
    result = cbrkit.retrieval.apply(casebase, query, retriever)

    assert len(result.similarities) == 5
    assert len(result.ranking) == 5
    assert len(result.casebase) == 5


def test_retrieve_nested():
    query_name = 42
    casebase_file = "data/cars-1k.yaml"

    casebase: dict[int, Any] = cbrkit.loaders.yaml(casebase_file)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "model": cbrkit.sim.attribute_value(
                    attributes={
                        "make": cbrkit.sim.strings.levenshtein(),
                        "manufacturer": cbrkit.sim.strings.taxonomy.load(
                            "./data/cars-taxonomy.yaml",
                            measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
                        ),
                    }
                ),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        min_similarity=0.5,
    )
    result = cbrkit.retrieval.apply(casebase, query, retriever)

    assert len(casebase) == 999
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name

    model_sim = result.similarities[query_name].attributes["model"]

    assert isinstance(model_sim, cbrkit.sim.AttributeValueSim)
    assert model_sim.value == 1.0
    assert model_sim.attributes["make"] == 1.0
