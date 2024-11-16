"""
.. include:: ../cli.md
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

from cbrkit.helpers import unpack_sim

try:
    import typer
    from rich import print
except ModuleNotFoundError:
    print(
        "Please install cbrkit with the [cli] extra to use the command line interface."
    )
    raise

import cbrkit

app = typer.Typer()


@app.callback()
def app_callback():
    pass


class ParallelStrategy(str, Enum):
    queries = "queries"
    casebase = "casebase"


@app.command()
def retrieve(
    casebase_path: Path,
    queries_path: Path,
    retriever: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    print_ranking: bool = True,
    print_similarities: bool = False,
    output_path: Path | None = None,
    processes: int = 1,
    parallel: ParallelStrategy = ParallelStrategy.queries,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    retrievers = cbrkit.retrieval.load(retriever)

    results = cbrkit.retrieval.mapply(
        casebase, queries, retrievers, processes, parallel.value
    )

    if output_path:
        results_dict = {
            query_name: result.as_dict() for query_name, result in results.items()
        }

        with output_path.with_suffix(".json").open("w") as fp:
            json.dump(results_dict, fp, indent=2)

    if print_ranking or print_similarities:
        for query_name, result in results.items():
            print(f"Query: {query_name}")

            if print_ranking:
                print(f"Ranking: {", ".join(map(str, result.ranking))}")

            if print_similarities:
                print("Similarities:")
                for case_name, similarity in result.similarities.items():
                    print(f"  {case_name}: {unpack_sim(similarity)}")

            print()


@app.command()
def serve(
    retriever: list[str],
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    root_path: str = "",
) -> None:
    import uvicorn

    sys.path.extend(str(x) for x in search_path)
    os.environ["CBRKIT_RETRIEVER"] = ",".join(retriever)

    uvicorn.run(
        "cbrkit.api:app",
        host=host,
        port=port,
        reload=reload,
        root_path=root_path,
    )


if __name__ == "__main__":
    app()
