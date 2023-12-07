from collections.abc import Callable, Collection
from pathlib import Path

from rich import print
from typer import Typer

import cbrkit

app = Typer()


@app.command()
def retrieve(casebase_path: Path, queries_path: Path, retriever: str):
    casebase: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(casebase_path)
    queries: cbrkit.model.Casebase[dict[str, str]] = cbrkit.load_path(queries_path)

    retriever_funcs: cbrkit.model.Retriever | Collection[
        cbrkit.model.Retriever
    ] = cbrkit.load.import_string(retriever)

    if not isinstance(retriever_funcs, Collection):
        retriever_funcs = [retriever_funcs]

    assert all(isinstance(func, Callable) for func in retriever_funcs)

    for query_name, query in queries.items():
        for retriever_idx, retriever_func in enumerate(retriever_funcs):
            result = retriever_func(casebase, query)

            print(f"{retriever_idx=}, {query_name=}")
            print(result.ranking)
            print()


if __name__ == "__main__":
    app()
