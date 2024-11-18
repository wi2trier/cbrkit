import polars as pl

import cbrkit


def test_simple():
    df = pl.read_csv("data/cars-1k.csv")
    casebase = cbrkit.loaders.polars(df)

    query = {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }

    sim_func = cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
            "manufacturer": cbrkit.sim.strings.taxonomy.load(
                "./data/cars-taxonomy.yaml",
                measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
            ),
            "make": cbrkit.sim.strings.levenshtein(),
            "miles": cbrkit.sim.numbers.linear(max=100000),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )

    retriever = cbrkit.retrieval.build(sim_func, limit=5)
    retrieval_result = cbrkit.retrieval.apply(casebase, query, retriever)

    reuse_func = cbrkit.reuse.build(
        adaptation_func=cbrkit.adapt.attribute_value(
            attributes={
                "price": cbrkit.adapt.numbers.aggregate("mean"),
                "make": cbrkit.adapt.strings.regex("a[0-9]", "a[0-9]", "a4"),
                "manufacturer": cbrkit.adapt.strings.regex(
                    "audi", "audi", "mercedes-benz"
                ),
                "miles": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        similarity_func=sim_func,
    )

    reuse_result = cbrkit.reuse.apply(retrieval_result.casebase, query, reuse_func)

    assert len(retrieval_result.casebase) == len(reuse_result.casebase)
