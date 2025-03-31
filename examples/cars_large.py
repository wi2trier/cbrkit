import pandas as pd

import cbrkit

casebase_file = "data/cars-10m.csv"
df = pd.read_csv(casebase_file)

casebase = cbrkit.loaders.pandas(df)
queries = cbrkit.loaders.pandas(df.iloc[:100])

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
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        multiprocessing=True,
    ),
    limit=5,
)

result = cbrkit.retrieval.apply_queries(
    casebase,
    queries,
    retriever,
)

assert len(result.ranking) == 5
