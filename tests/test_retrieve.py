from typing import Any

import pandas as pd

import cbrkit


def test_retrieve_pandas():
    query_name = 42
    casebase_file = "data/cars-1k.csv"

    df = pd.read_csv(casebase_file)
    casebase = cbrkit.loaders.dataframe(df)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.build(
        cbrkit.global_sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numeric.linear(max=100000),
                "year": cbrkit.sim.numeric.linear(max=50),
                "manufacturer": cbrkit.sim.taxonomy.load(
                    "./data/cars-taxonomy.yaml",
                    measure=cbrkit.sim.taxonomy.wu_palmer(),
                ),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numeric.linear(max=1000000),
            },
            types_fallback=cbrkit.sim.generic.equality(),
            aggregator=cbrkit.global_sim.aggregator(pooling="mean"),
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


def test_retrieve_nested():
    query_name = 42
    casebase_file = "data/cars-1k.yaml"

    casebase: dict[int, Any] = cbrkit.loaders.yaml(casebase_file)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.build(
        cbrkit.global_sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numeric.linear(max=100000),
                "year": cbrkit.sim.numeric.linear(max=50),
                "model": cbrkit.global_sim.attribute_value(
                    attributes={
                        "make": cbrkit.sim.strings.levenshtein(),
                        "manufacturer": cbrkit.sim.taxonomy.load(
                            "./data/cars-taxonomy.yaml",
                            measure=cbrkit.sim.taxonomy.wu_palmer(),
                        ),
                    }
                ),
            },
            aggregator=cbrkit.global_sim.aggregator(pooling="mean"),
        ),
        limit=5,
    )
    result = cbrkit.retrieval.apply(casebase, query, retriever)

    assert len(casebase) == 999
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name

    model_sim = result.similarities[query_name].by_attribute["model"]

    assert isinstance(model_sim, cbrkit.global_sim.AttributeValueSim)
    assert model_sim.value == 1.0
    assert model_sim.by_attribute["make"] == 1.0
