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
        retriever_func=retriever,
    )

    result = cbrkit.cycle.apply_queries(casebase, {"default": query}, retriever, reuser)

    assert len(result.retrieval.casebase) == len(result.reuse.casebase)
