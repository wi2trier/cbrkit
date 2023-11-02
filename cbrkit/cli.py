from pathlib import Path

import pandas as pd
from rich import print
from typer import Typer

import cbrkit

app = Typer()


@app.command()
def retrieve(path: Path):
    if path.suffix == ".csv":
        casebase = cbrkit.load_dataframe(pd.read_csv(path))
    else:
        casebase = cbrkit.load_file(path)

    result = cbrkit.retrieve(
        casebase,
        query=casebase[0],
        similarity_func=lambda x, y: x.equals(y),
        casebase_limit=5,
    )

    print(result)


if __name__ == "__main__":
    app()
