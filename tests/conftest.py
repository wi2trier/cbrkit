"""Shared fixtures and helpers for cbrkit tests."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytest

import cbrkit


def _custom_numeric_sim(x: float, y: float) -> float:
    return 1 - abs(x - y) / 100000


@dataclass(slots=True, frozen=True)
class SearchingRetriever(cbrkit.typing.CbrFunc[int, str, float]):
    """Retriever that searches its own cases and ignores the given ones, like a storage retriever."""

    cases: Mapping[int, str]

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[int, str], str]],
        /,
    ) -> Sequence[
        tuple[cbrkit.typing.Casebase[int, str], cbrkit.typing.SimMap[int, float]]
    ]:
        return [(self.cases, dict.fromkeys(self.cases, 1.0)) for _ in batches]


# --- Session-scoped fixtures ---


@pytest.fixture(scope="session")
def cars_csv_casebase() -> Mapping[int, Any]:
    """Cars casebase loaded from CSV via polars."""
    df = pl.read_csv("data/cars-1k.csv")
    return cbrkit.io.Polars(df)


@pytest.fixture(scope="session")
def cars_yaml_casebase() -> Mapping[int, Any]:
    """Cars casebase loaded from YAML."""
    return cbrkit.io.load("data/cars-1k.yaml")


@pytest.fixture(scope="session")
def small_casebase() -> dict[int, dict[str, int]]:
    """Small 2-entry casebase for retain/revise tests."""
    return {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
    }


@pytest.fixture(scope="session")
def medium_casebase() -> dict[int, dict[str, int]]:
    """Medium 3-entry casebase extending small_casebase."""
    return {
        0: {"price": 12000, "year": 2008},
        1: {"price": 9000, "year": 2012},
        2: {"price": 11000, "year": 2010},
    }


@pytest.fixture(scope="session")
def car_query() -> dict[str, int | str]:
    """Full car query with all typical attributes."""
    return {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }


@pytest.fixture(scope="session")
def simple_query() -> dict[str, int]:
    """Simple query with price and year only."""
    return {"price": 10000, "year": 2010}


@pytest.fixture(scope="session")
def sim_func_simple() -> cbrkit.typing.AnySimFunc[Any, Any]:
    """Simple similarity function for price+year attributes."""
    return cbrkit.sim.AttributeValue(
        attributes={
            "price": cbrkit.sim.numbers.Linear(max_distance=100000),
            "year": cbrkit.sim.numbers.Linear(max_distance=50),
        },
        aggregator=cbrkit.sim.Aggregator(),
    )
