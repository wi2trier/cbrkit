# uv run examples/cars_large.py
"""Retrieves from a large tabular case base, scored column by column and spread over processes."""

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from typing import Any

import polars as pl

import cbrkit


def main(rows: int, executor: Executor | None = None) -> Mapping[Any, Sequence[Any]]:
    """Ranks the five most similar cars of `rows` cars for ten queries."""
    cars = pl.read_csv("data/cars-1k.csv")
    df = pl.concat([cars] * (rows // cars.height + 1)).head(rows)
    casebase = cbrkit.io.Polars(df)
    queries = cbrkit.io.Polars(df.head(10))
    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.strings.Levenshtein(),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        executor=executor,
        limit=5,
    )

    result = cbrkit.cycle.run(casebase, queries, retriever)

    return {query: entry.ranking for query, entry in result.items()}


if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        print(main(1_000_000, executor))
