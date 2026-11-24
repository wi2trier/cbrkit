"""The examples run end to end, with a stub language model instead of a remote one."""

import runpy
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"


def example(name: str) -> Callable[..., Any]:
    return runpy.run_path(str(EXAMPLES / f"{name}.py"))["main"]


def test_cars_large():
    with ThreadPoolExecutor(2) as executor:
        rankings = example("cars_large")(2000, executor)

    assert len(rankings) == 10
    assert all(len(ranking) == 5 for ranking in rankings.values())


def test_cars_system(tmp_path: Path):
    pytest.importorskip("fastmcp")
    casebase = tmp_path / "cars.csv"
    shutil.copy("data/cars-1k.csv", casebase)
    example("cars_system")(casebase)

    assert (tmp_path / "cars.csv.cbrkit.json").exists()
