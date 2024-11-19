"""
.. include:: ../../cli.md
"""

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

try:
    import typer
    from rich import print
except ModuleNotFoundError:
    print("Please install with the [cli] extra to use the command line interface.")
    raise

import cbrkit

__all__ = ["app"]

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
    retrievers: list[cbrkit.typing.RetrieverFunc] = cbrkit.helpers.load_callables(
        retriever
    )

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
                    print(f"  {case_name}: {cbrkit.helpers.unpack_sim(similarity)}")

            print()


@app.command()
def reuse(
    casebase_path: Path,
    queries_path: Path,
    reuser: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    output_path: Path | None = None,
    processes: int = 1,
    parallel: ParallelStrategy = ParallelStrategy.queries,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    reusers: list[cbrkit.typing.ReuserFunc] = cbrkit.helpers.load_callables(reuser)

    results = cbrkit.reuse.mapply(casebase, queries, reusers, processes, parallel.value)

    if output_path:
        results_dict = {
            query_name: result.as_dict() for query_name, result in results.items()
        }

        with output_path.with_suffix(".json").open("w") as fp:
            json.dump(results_dict, fp, indent=2)


@app.command()
def serve(
    retriever: Annotated[list[str], typer.Option(default_factory=list)],
    reuser: Annotated[list[str], typer.Option(default_factory=list)],
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    host: str = "0.0.0.0",
    port: int = 8080,
    reload: bool = False,
    root_path: str = "",
) -> None:
    import uvicorn

    sys.path.extend(str(x) for x in search_path)
    os.environ["CBRKIT_RETRIEVER"] = ",".join(retriever)
    os.environ["CBRKIT_REUSER"] = ",".join(reuser)

    uvicorn.run(
        "cbrkit.api:app",
        host=host,
        port=port,
        reload=reload,
        root_path=root_path,
    )


if __name__ == "__main__":
    app()
