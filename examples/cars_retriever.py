import cbrkit

retriever = cbrkit.retrieval.build(
    cbrkit.sim.attribute_value(
        attributes={
            "year": cbrkit.sim.numbers.linear(max=50),
            "make": cbrkit.sim.strings.levenshtein(),
            "miles": cbrkit.sim.numbers.linear(max=1000000),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    ),
    limit=5,
)
