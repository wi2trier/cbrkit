# uv run -m cbrkit retrieve data/cars-1k.csv data/cars-queries.csv examples.cars_retriever:retriever --output-path data/output.json
import cbrkit

retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "year": cbrkit.sim.numbers.linear(max=50),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numbers.linear(max=1000000),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
    ),
    limit=5,
)
