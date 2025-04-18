# uv run examples/cars_rag.py
import polars as pl

import cbrkit
from pydantic import BaseModel

df = pl.read_csv("data/cars-1k.csv").head(10)
casebase = cbrkit.loaders.polars(df)

sim_func = cbrkit.sim.attribute_value(
    attributes={
        "year": cbrkit.sim.numbers.linear(max=50),
        "make": cbrkit.sim.strings.levenshtein(),
        "miles": cbrkit.sim.numbers.linear(max=1000000),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)

retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(sim_func),
    limit=5,
)


class Response(BaseModel):
    car: str
    price: float
    summary: str


synthesizer = cbrkit.synthesis.build(
    cbrkit.synthesis.providers.anthropic(
        model="claude-3-haiku-20240307", response_type=Response, max_tokens=200
    ),
    cbrkit.synthesis.prompts.default(
        "Give me a summary of the found cars.",
        metadata=cbrkit.helpers.get_metadata(sim_func),
    ),
)

retrieval = cbrkit.retrieval.apply_query(
    casebase,
    casebase[0],
    retriever,
)
response = cbrkit.synthesis.apply_result(retrieval, synthesizer).response

print(response)

# Exemplary output:
# The query details a 2011 Subaru Outback, priced at $13,686 with a diesel engine and manual transmission. It has a notable mileage of 6,024,800 miles, a clean title, rear-wheel drive, and is painted black.
#
# Among the retrieved cases:
#
# 1. **Exact Match**: A 2011 Subaru Outback with identical attributes to the queried car, confirming perfect similarity.
#
# 2. **Close Comparisons**:
#    - A 2006 Skoda Octavia, priced significantly lower at $462, with similar high mileage but runs on gas. It also maintains a clean title and manual transmission, but offers four-wheel drive.
#    - Another 2006 Skoda Octavia for $3,145 shares manual transmission and black color, but runs on gas and includes slight differences in drive and title status.
#
# 3. **Other Notable Mentions**:
#    - A 2006 Skoda Superb and 2006 Nissan Patrol both feature similar specifications, mainly in terms of manual transmission and compact type, although differing in fuel type and certain attributes like drive type and title status.
#
# Overall, while the exact match shares the same specifics as the queried Subaru, other retrieved cases show variations primarily in age, fuel type, and price.
