import polars as pl
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

import cbrkit

df = pl.read_csv("data/cars-1k.csv").head(30)

casebase = cbrkit.loaders.polars(df)

sim_func = cbrkit.sim.attribute_value(
    attributes={
        "year": cbrkit.sim.numbers.linear(max=50),
        "make": cbrkit.sim.strings.levenshtein(),
        "miles": cbrkit.sim.numbers.linear(max=1000000),
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)

retriever = cbrkit.retrieval.build(sim_func)

prompt = cbrkit.synthesis.prompts.default(
    instructions="Give me a summary of the aptitude for daily driving each of the retrieved cars.",
    # metadata=cbrkit.helpers.get_metadata(sim_func),
)
provider = cbrkit.synthesis.providers.pydantic_ai(
    Agent(
        OpenAIChatModel("gpt-5.1-codex"),
        output_type=str,
    ),
    deps=None,
)
synthesizer = cbrkit.synthesis.build(
    provider,
    prompt,
)
queries = [casebase[i] for i in range(2)]

retrievals = [
    cbrkit.retrieval.apply_query(casebase, query, retriever) for query in queries
]

# batches are tuples of casebase, query, and retrieval similarities
batches = [
    (casebase, query, retrieval.similarities)
    for query, retrieval in zip(queries, retrievals, strict=True)
]

pooling_prompt = cbrkit.synthesis.prompts.pooling(
    "Which of the following cars is the best replacement for the queried cars?"
)
pooling_func = cbrkit.synthesis.pooling(provider, pooling_prompt)
get_result = cbrkit.synthesis.chunks(synthesizer, pooling_func, size=10)
response = get_result(batches)
print("Response:")
print(response)

# Exemplary output:
# [ " ## Car Search Results\n\n   Here are the search results for your query:\n\n   1. Year: 2011, Make: Mercedes-Benz, Model: Viano, Transmission: Manual, Drivetrain: Front-wheel drive, Color: Black, Price: $22,168, Condition: Rebuilt\n\n   Other options:\n\n   1. Year: 2011, Make: Ford, Model: S-Max, Type: Diesel van, Transmission: Manual, Drivetrain: Front-wheel drive, Price: $9,437\n\n   2. Year: 2012, Make: Chrysler, Model: Town-Country, Type: Van, Transmission: Manual, Drivetrain: Front-wheel drive, Price: $1,846\n\n   3. Year: 2006, Make: Fiat, Model: Doblo, Type: Diesel van, Transmission: Manual, Drivetrain: All-wheel drive, Price: $3,515\n\n   4. Year: 2002, Make: Hyundai, Model: Matrix, Type: Van, Fuel: Gasoline, Transmission: Manual, Drivetrain: Front-wheel drive, Price: $1,073", " ## Vehicle Analysis based on Given List\n\nBased on the provided list, here is an analysis of each vehicle in terms of their similarity to a query car (a diesel-powered forward driving van) for daily driving aptitude:\n\n1. **2011 Mercedes-Benz Viano (rank 1):** This vehicle shares the most similar characteristics with the query car, as it is a diesel-powered forward driving van and has a manual transmission. Its high mileage might be a concern but it is the best match based on the given criteria.\n\n2. **2011 Ford S-Max (rank 2):** Although it is also a diesel-powered forward driving van, differences in manufacturer, model, and mileage make it less similar to the query car. However, it could still be considered as an option if the other factors are less important.\n\n3. **2006 Fiat Doblo (rank 3):** This vehicle is a diesel-powered van but has a 4WD system instead of FWD like the query car, making it less suitable for daily driving compared to options 1 and 2.\n\n4. **2008 SEAT Alhambra (rank 4):** This vehicle is a diesel-powered 4WD van, which makes it even less suitable for daily driving as a van compared to the other options in this list due to its all-wheel drive system.\n\n5. **2002 Hyundai Matrix (rank 5):** This vehicle is not a van but a small SUV/station wagon, has a gas engine instead of diesel and front-wheel drive. It is the least suitable option for daily driving as a van compared to all other options in this list.\n\nIn summary, the top three vehicles that might be considered for daily driving aptitude are:\n1. 2011 Mercedes-Benz Viano (rank 1)\n2. 2011 Ford S-Max (rank 2)\n3. 2006 Fiat Doblo (rank 3), although it is less suitable due to its 4WD system compared to the other two options.", "1. **Car 1 (2011 Mercedes-Benz Viano):** Similar to another Mercedes-Benz Viano from 2013, both are diesel vehicles with rebuilt title status and black in color. Given its type (van) and drive (front-wheel drive), this car seems suitable for daily driving.\n  2. **Car 2 (2012 Chrysler Town-Country):** Different manufacturer, make, fuel type (gas), and drive type. Might not be as suitable for daily driving compared to the Mercedes-Benz Viano.\n  3. **Car 3 (2006 Fiat Doblo):** Different manufacturer, make, fuel type (diesel), drive type (4WD), and paint color. Might not be as suitable for daily driving compared to the Mercedes-Benz Viano.\n  4. **Car 4 (2008 SEAT Alhambra):** Different manufacturer, make, fuel type (diesel), drive type (4WD), and paint color. Might not be as suitable for daily driving compared to the Mercedes-Benz Viano.\n  5. **Car 5 (2002 Hyundai Matrix):** Different manufacturer, make, fuel type (gas), title status (rebuilt), and paint color. Might not be as suitable for daily driving compared to the Mercedes-Benz Viano.\n\nIn conclusion, the 2011 Mercedes-Benz Viano appears to be the most suitable option for daily driving among the provided cars based on their similarities. However, it is crucial to verify all details before making a final decision.", " Based on the information provided, here are the top 5 cars that are most similar to the query car (Mercedes-Benz Viano) in terms of make, manufacturer, fuel type, and color:\n\n1. The first car: Mercedes-Benz Viano from the year 2011 with manual transmission and front-wheel drive. It has a rebuilt title status, diesel fuel, 203593 miles on it, and is painted black. This car seems to be quite similar to the query car in terms of make, manufacturer, fuel type, color, and drive type, but it has a higher mileage and is from a slightly older year.\n\n2. The second car: Chrysler Town-Country from the year 2012 with manual transmission and front-wheel drive. It has a clean title status, gas fuel, 122800 miles on it, and is also painted black. This car doesn't match the query car in terms of make, manufacturer, fuel type, or drive type, but it is from the same year and color, and it has a lower mileage.\n\nAmong these two cars, the first one is more similar as it matches the year and drive type (manual transmission and front-wheel drive) as well, although it has a higher mileage. The second car doesn't match the drive type or year, but it has a lower mileage." ]
