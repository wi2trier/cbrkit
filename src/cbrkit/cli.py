"""
.. include:: ../../cli.md
"""

import os
import sys
from pathlib import Path
from typing import Annotated

import orjson

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "cli"):
    import typer
    from rich import print


__all__ = ["app"]

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback()
def app_callback():
    pass


@app.command()
def retrieve(
    casebase_path: Path,
    queries_path: Path,
    retriever: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    print_ranking: bool = True,
    print_similarities: bool = False,
    output_path: Path | None = None,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    retrievers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]] = (
        cbrkit.helpers.load_callables(retriever)
    )

    result = cbrkit.retrieval.apply_queries(casebase, queries, retrievers)

    if output_path:
        cbrkit.dumpers.file(output_path, result)

    if print_ranking or print_similarities:
        for query_key, query_result in result.final_step.queries.items():
            print(f"Query: {query_key}")

            if print_ranking:
                print(f"Ranking: {', '.join(map(str, query_result.ranking))}")

            if print_similarities:
                print("Similarities:")
                for case_name, similarity in query_result.similarities.items():
                    print(f"  {case_name}: {cbrkit.helpers.unpack_float(similarity)}")

            print()


@app.command()
def reuse(
    casebase_path: Path,
    queries_path: Path,
    reuser: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    output_path: Path | None = None,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    reusers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = (
        cbrkit.helpers.load_callables(reuser)
    )

    result = cbrkit.reuse.apply_queries(casebase, queries, reusers)

    if output_path:
        cbrkit.dumpers.file(output_path, result)


@app.command()
def cycle(
    casebase_path: Path,
    queries_path: Path,
    retriever: str,
    reuser: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    output_path: Path | None = None,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    retrievers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]] = (
        cbrkit.helpers.load_callables(retriever)
    )
    reusers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = (
        cbrkit.helpers.load_callables(reuser)
    )

    result = cbrkit.cycle.apply_queries(casebase, queries, retrievers, reusers)

    if output_path:
        cbrkit.dumpers.file(output_path, result)


@app.command()
def synthesis(
    casebase_path: Path,
    queries_path: Path,
    retriever: str,
    reuser: str,
    synthesizer: str,
    search_path: Annotated[list[Path], typer.Option(default_factory=list)],
    output_path: Path | None = None,
) -> None:
    sys.path.extend(str(x) for x in search_path)
    casebase = cbrkit.loaders.path(casebase_path)
    queries = cbrkit.loaders.path(queries_path)
    retrievers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]] = (
        cbrkit.helpers.load_callables(retriever)
    )
    reusers: list[cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = (
        cbrkit.helpers.load_callables(reuser)
    )
    synthesis_func: cbrkit.typing.MaybeFactory[cbrkit.typing.SynthesizerFunc] = (
        cbrkit.helpers.load_callable(synthesizer)
    )

    cycle_result = cbrkit.cycle.apply_queries(casebase, queries, retrievers, reusers)
    synthesis_result = cbrkit.synthesis.apply_result(
        cycle_result.final_step, synthesis_func
    )

    if output_path:
        cbrkit.dumpers.file(output_path, synthesis_result)


@app.command()
def serve(
    retriever: Annotated[list[str], typer.Option(default_factory=list)],
    reuser: Annotated[list[str], typer.Option(default_factory=list)],
    synthesizer: Annotated[list[str], typer.Option(default_factory=list)],
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
    os.environ["CBRKIT_SYNTHESIZER"] = ",".join(synthesizer)

    uvicorn.run(
        "cbrkit.api:app",
        host=host,
        port=port,
        reload=reload,
        root_path=root_path,
    )


@app.command()
def openapi(file: Path | None = None):
    from cbrkit.api import app

    schema = orjson.dumps(
        app.openapi(),
        option=orjson.OPT_INDENT_2,
    )

    if file is None:
        print(schema.decode())

    else:
        print(f"Writing OpenAPI schema to {file}")

        with file.open("wb") as fp:
            fp.write(schema)


if __name__ == "__main__":
    app()
