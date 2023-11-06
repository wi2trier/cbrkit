from pathlib import Path

from rich import print
from typer import Typer

import cbrkit

app = Typer()


@app.command()
def retrieve(path: Path):
    casebase: cbrkit.model.Casebase[dict[str, dict[str, str]]] = cbrkit.load_path(path)

    result = cbrkit.retrieve(
        casebase,
        query=casebase["42"],
        similarity_func="equality",
        casebase_limit=5,
    )

    print(result.ranking)


if __name__ == "__main__":
    app()
