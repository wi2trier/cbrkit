# uv run -m cbrkit retrieve data/cars-1k.csv data/cars-queries.csv examples.cars_retriever:retriever --output-path data/output.json
# uv run -m cbrkit serve --retriever examples.cars_retriever:retriever
# curl --location 'localhost:8080/retrieve' \
# --header 'Content-Type: application/json' \
# --data '{
#     "casebase": "PATH_TO_CASEBASE.csv",
#     "queries": "PATH_TO_QUERIES.csv"
# }'
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
