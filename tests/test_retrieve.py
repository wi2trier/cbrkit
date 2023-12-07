import pandas as pd

import cbrkit

query_name = 42
casebase_file = "data/cars-1k.csv"


# TODO: Create some taxonomy similarity measure
def test_retrieve_pandas():
    df = pd.read_csv(casebase_file)
    casebase = cbrkit.load_dataframe(df)
    query = casebase[query_name]
    retrieve = cbrkit.retriever(
        cbrkit.case_sim.factories.by_attributes(
            {
                "manufacturer": cbrkit.data_sim.strings.levenshtein(),
                "miles": cbrkit.data_sim.numeric.linear(max=1000000),
            }
        ),
        casebase_limit=5,
    )
    result = retrieve(casebase, query)

    assert len(casebase) == 999  # csv contains header
    assert len(result.ranking) == len(casebase)
    assert len(result.casebase) == 5
    assert result.ranking[0] == query_name
