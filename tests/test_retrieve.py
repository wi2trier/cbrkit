import cbrkit


def test_retrieve():
    casebase: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(
        "data/cars-1k.csv"
    )
    query_name = "42"
    query = casebase[query_name]

    result = cbrkit.retrieve(
        casebase,
        query=query,
        similarity_func="equality",
        casebase_limit=5,
    )

    assert len(casebase) == 999  # csv contains header
    assert len(result.ranking) == len(casebase)
    assert len(result.casebase) == 5
    assert result.ranking[0] == query_name
