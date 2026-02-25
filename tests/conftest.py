"""Shared fixtures and helpers for cbrkit tests."""

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import polars as pl
import pytest

import cbrkit
from cbrkit.retrieval.indexable import resolve_casebases


def _custom_numeric_sim(x: float, y: float) -> float:
    return 1 - abs(x - y) / 100000


class FakeIndexable(
    cbrkit.typing.IndexableFunc[Mapping[int, str], Collection[int]],
):
    """Fake indexable implementing create/update/delete for testing."""

    def __init__(self) -> None:
        self._data: dict[int, str] | None = None

    @property
    def index(self) -> Mapping[int, str]:
        if self._data is None:
            return {}
        return self._data

    def has_index(self) -> bool:
        return self._data is not None

    def create_index(self, data: Mapping[int, str]) -> None:
        self._data = dict(data)

    def update_index(self, data: Mapping[int, str]) -> None:
        if self._data is None:
            self.create_index(data)
            return
        self._data.update(data)

    def delete_index(self, data: Collection[int]) -> None:
        if self._data is None:
            return
        for key in data:
            self._data.pop(key, None)


class FakeIndexableRetriever(
    cbrkit.typing.RetrieverFunc[int, str, float],
    cbrkit.typing.IndexableFunc[Mapping[int, str], Collection[int]],
):
    """Fake retriever with indexable support for testing."""

    def __init__(self) -> None:
        self._indexed_casebase: dict[int, str] | None = None

    @property
    def index(self) -> Mapping[int, str]:
        if self._indexed_casebase is None:
            return {}
        return self._indexed_casebase

    def has_index(self) -> bool:
        return self._indexed_casebase is not None

    def create_index(self, data: Mapping[int, str]) -> None:
        self._indexed_casebase = dict(data)

    def update_index(self, data: Mapping[int, str]) -> None:
        if self._indexed_casebase is None:
            self.create_index(data)
            return

        self._indexed_casebase.update(data)

    def delete_index(self, data: Collection[int]) -> None:
        if self._indexed_casebase is None:
            return

        for key in data:
            self._indexed_casebase.pop(key, None)

    def __call__(
        self,
        batches: Sequence[tuple[cbrkit.typing.Casebase[int, str], str]],
    ) -> Sequence[
        tuple[cbrkit.typing.Casebase[int, str], cbrkit.typing.SimMap[int, float]]
    ]:
        return [
            (dict(casebase), {key: 1.0 for key in casebase})
            for casebase, _query in resolve_casebases(batches, self._indexed_casebase)
        ]


# --- Session-scoped fixtures ---


@pytest.fixture(scope="session")
def cars_csv_casebase() -> Mapping[int, Any]:
    """Cars casebase loaded from CSV via polars."""
    df = pl.read_csv("data/cars-1k.csv")
    return cbrkit.loaders.polars(df)


@pytest.fixture(scope="session")
def cars_yaml_casebase() -> Mapping[int, Any]:
    """Cars casebase loaded from YAML."""
    return cbrkit.loaders.path("data/cars-1k.yaml")


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
def sim_func_simple() -> cbrkit.typing.AnySimFunc:
    """Simple similarity function for price+year attributes."""
    return cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )
