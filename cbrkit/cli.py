from pathlib import Path

from rich import print
from typer import Typer

import cbrkit

app = Typer()


@app.command()
def retrieve(casebase_path: Path, queries_path: Path):
    casebase: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(casebase_path)
    queries: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(queries_path)

    for query_name, query in queries.items():
        result = cbrkit.retrieve(
            casebase,
            query=query,
            similarity_func="equality",
            casebase_limit=5,
        )

        print(f"Query: {query_name}")
        print(result.ranking)
        print()


if __name__ == "__main__":
    app()
