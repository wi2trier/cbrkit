# uv run --with uvicorn examples/cars_system.py
"""Serves the cars case base over REST and MCP after inferring its sidecar once."""

from pathlib import Path

from fastapi import FastAPI

from cbrkit import io, spec
from cbrkit.system import System
from cbrkit.system.api import app


def main(casebase: Path) -> FastAPI:
    """The REST and MCP app of a case base, inferring and writing its sidecar if there is none."""
    if not io.companion(casebase, "sidecar").exists():
        source = io.detect(casebase)
        config = spec.infer(io.load(casebase, source), source)
        spec.write(io.companion(casebase, "sidecar"), config)

    return app(System.load(casebase), mcp=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(main(Path("data/cars-1k.csv")))
