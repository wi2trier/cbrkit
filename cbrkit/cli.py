from pathlib import Path

from rich import print
from typer import Typer

import cbrkit

app = Typer()


@app.command()
def retrieve(path: Path):
    casebase = cbrkit.load_path(path)

    result = cbrkit.retrieve(
        casebase,
        query=casebase[0],
        similarity_func=lambda x, y: x.equals(y),
        casebase_limit=5,
    )

    print(result)


if __name__ == "__main__":
    app()
