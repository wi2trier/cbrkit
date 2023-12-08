"""
.. include:: ../cli.md
"""


from pathlib import Path

try:
    from rich import print
    from typer import Typer
except ModuleNotFoundError:
    print(
        "Please install cbrkit with the [cli] extra to use the command line interface."
    )
    raise

import cbrkit
from cbrkit.retrieve import import_retrievers

app = Typer()


@app.command()
def retrieve(casebase_path: Path, queries_path: Path, retriever: str):
    casebase = cbrkit.load.path(casebase_path)
    queries = cbrkit.load.path(queries_path)
    retrievers = import_retrievers(retriever)

    for query_name, query in queries.items():
        result = cbrkit.retrieve(casebase, query, retrievers)
        print(f"Query: {query_name}")
        print(result.ranking)
        print()


if __name__ == "__main__":
    app()
