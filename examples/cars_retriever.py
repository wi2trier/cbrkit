# uv run -m cbrkit retrieve data/cars-1k.csv data/cars-queries.csv examples.cars_retriever:retriever --output-path data/output.json
# uv run -m cbrkit serve --retriever examples.cars_retriever:retriever
# curl --location "localhost:8080/retrieve" --form casebase="/Users/mlenz/Developer/wi2trier/cbrkit/data/cars-1k.csv" --form queries="/Users/mlenz/Developer/wi2trier/cbrkit/data/cars-queries.csv"
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
