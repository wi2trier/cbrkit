import pandas as pd

import cbrkit

query_name = 42
casebase_file = "data/cars-1k.csv"


def test_retrieve_pandas():
    df = pd.read_csv(casebase_file)
    casebase = cbrkit.load.dataframe(df)
    query = casebase[query_name]
    retriever = cbrkit.retriever(
        cbrkit.sim.factories.tabular(
            attributes={
                "price": cbrkit.sim.numeric.linear(max=100000),
                "year": cbrkit.sim.numeric.linear(max=2020, min=1960),
                "manufacturer": cbrkit.sim.strings.taxonomy(
                    "./data/cars-taxonomy.yaml", measure="wu_palmer"
                ),
                # TODO: needs nlp extra to be available during tests
                # "make": cbrkit.data_sim.strings.levenshtein(),
                "miles": cbrkit.sim.numeric.linear(max=1000000),
            },
            types_fallback=cbrkit.sim.generic.equality(),
            aggregator=cbrkit.sim.helpers.aggregator(pooling="mean"),
        ),
        casebase_limit=5,
    )
    result = cbrkit.retrieve(casebase, query, retriever)

    assert len(casebase) == 999  # csv contains header
    assert len(result.ranking) == len(casebase)
    assert len(result.casebase) == 5
    assert result.ranking[0] == query_name
