import pandas as pd

import cbrkit

query_name = 42
casebase_file = "data/cars-1k.csv"


def test_retrieve_default():
    casebase: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(casebase_file)
    query = casebase[query_name]

    result = cbrkit.retrieve(
        casebase,
        query=query,
        similarity_func="datatypes",
        casebase_limit=5,
    )

    assert len(casebase) == 999  # csv contains header
    assert len(result.ranking) == len(casebase)
    assert len(result.casebase) == 5
    assert result.ranking[0] == query_name


# TODO: Create some taxonomy similarity measure
custom_sim_func = cbrkit.case_sim.factories.by_attributes(
    {
        "manufacturer": cbrkit.data_sim.strings.levenshtein(),
        "miles": cbrkit.data_sim.numeric.linear(max=1000000),
    },
    aggregate=cbrkit.case_sim.aggregate(),
)


# TODO: Pandas dataframe is indexed by int, but should use strings instead!
def test_retrieve_custom():
    df = pd.read_csv(casebase_file)
    casebase = cbrkit.load_dataframe(df)
    query = casebase[query_name]

    result = cbrkit.retrieve(
        casebase,
        query=query,
        similarity_func=custom_sim_func,
        casebase_limit=5,
    )

    assert len(casebase) == 999  # csv contains header
    assert len(result.ranking) == len(casebase)
    assert len(result.casebase) == 5
    assert result.ranking[0] == query_name
