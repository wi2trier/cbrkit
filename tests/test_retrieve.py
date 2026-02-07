from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest

import cbrkit
from cbrkit.retrieval.common import resolve_casebases
from cbrkit.typing import Casebase


def _custom_numeric_sim(x: float, y: float) -> float:
    return 1 - abs(x - y) / 100000


class FakeIndexableRetriever(
    cbrkit.typing.RetrieverFunc[int, str, float],
    cbrkit.typing.IndexableFunc[Casebase[int, str]],
):
    def __init__(self) -> None:
        self._indexed_casebase: dict[int, str] | None = None

    def index(self, casebase: Casebase[int, str], prune: bool = True) -> None:
        self._indexed_casebase = dict(casebase)

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


@pytest.mark.skip(reason="this test is slow on macOS")
def test_retrieve_multiprocessing():
    query_name = 42
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "manufacturer": cbrkit.sim.taxonomy.build(
                        "./data/cars-taxonomy.yaml",
                        cbrkit.sim.taxonomy.wu_palmer(),
                    ),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": _custom_numeric_sim,
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
            multiprocessing=2,
            chunksize=5,
        )
    )
    result = cbrkit.retrieval.apply_query(
        casebase,
        casebase[query_name],
        retriever,
    )

    assert len(result.ranking) == 999


def test_retrieve_dataframe():
    query_name = 42
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "manufacturer": cbrkit.sim.taxonomy.build(
                        "./data/cars-taxonomy.yaml",
                        cbrkit.sim.taxonomy.wu_palmer(),
                    ),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=5,
    )
    result = cbrkit.retrieval.apply_query(
        casebase,
        query,
        retriever,
    )

    assert len(casebase) == 999  # csv contains header
    assert len(result.similarities) == 5
    assert len(result.ranking) == 5
    assert len(result.casebase) == 5
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name


def test_retrieve_dataframe_custom_query():
    casebase_file = "data/cars-1k.csv"

    df = pl.read_csv(casebase_file)
    casebase = cbrkit.loaders.polars(df)

    query = {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }

    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "manufacturer": cbrkit.sim.taxonomy.build(
                        "./data/cars-taxonomy.yaml",
                        cbrkit.sim.taxonomy.wu_palmer(),
                    ),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": _custom_numeric_sim,
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=5,
    )
    result = cbrkit.retrieval.apply_query(
        casebase,
        query,
        retriever,
    )

    assert len(result.similarities) == 5
    assert len(result.ranking) == 5
    assert len(result.casebase) == 5


def test_retrieve_nested():
    query_name = 42
    casebase_file = "data/cars-1k.yaml"

    casebase: Mapping[int, Any] = cbrkit.loaders.path(casebase_file)
    query = casebase[query_name]
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "model": cbrkit.sim.attribute_value(
                        attributes={
                            "make": cbrkit.sim.strings.levenshtein(),
                            "manufacturer": cbrkit.sim.taxonomy.build(
                                "./data/cars-taxonomy.yaml",
                                cbrkit.sim.taxonomy.wu_palmer(),
                            ),
                        }
                    ),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        min_similarity=0.5,
    )
    result = cbrkit.retrieval.apply_query(
        casebase,
        query,
        retriever,
    )

    assert len(casebase) == 999
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name

    model_sim = result.similarities[query_name].attributes["model"]

    assert isinstance(model_sim, cbrkit.sim.AttributeValueSim)
    assert model_sim.value == 1.0
    assert model_sim.attributes["make"] == 1.0


def test_retrieve_indexed_casebase_resolution() -> None:
    from frozendict import frozendict

    retriever = FakeIndexableRetriever()
    retriever.index(frozendict({1: "a", 2: "b"}))

    result = cbrkit.retrieval.apply_query({}, "a", retriever)

    assert result.casebase[1] == "a"
    assert result.casebase[2] == "b"
    assert len(result.casebase) == 2


def test_retrieve_indexed_casebase_requires_index() -> None:
    retriever = FakeIndexableRetriever()

    with pytest.raises(ValueError, match="Call index\\(\\) first"):
        cbrkit.retrieval.apply_query({}, "a", retriever)


def test_retrieve_apply_query_indexed() -> None:
    from frozendict import frozendict

    retriever = FakeIndexableRetriever()
    retriever.index(frozendict({1: "a", 2: "b"}))

    result = cbrkit.retrieval.apply_query_indexed("a", retriever)

    assert result.casebase[1] == "a"
    assert result.casebase[2] == "b"
    assert len(result.casebase) == 2


def test_combine_sequence_indexed() -> None:
    from frozendict import frozendict

    r1 = FakeIndexableRetriever()
    r1.index(frozendict({1: "a", 2: "b"}))

    r2 = FakeIndexableRetriever()
    r2.index(frozendict({1: "a", 2: "b"}))

    retriever = cbrkit.retrieval.combine([r1, r2])
    result = cbrkit.retrieval.apply_query({}, "a", retriever)

    assert len(result.casebase) == 2
    assert result.casebase[1] == "a"
    assert result.casebase[2] == "b"


def test_combine_mapping_indexed() -> None:
    from frozendict import frozendict

    r1 = FakeIndexableRetriever()
    r1.index(frozendict({1: "a", 2: "b"}))

    r2 = FakeIndexableRetriever()
    r2.index(frozendict({1: "a", 2: "b"}))

    retriever = cbrkit.retrieval.combine({"r1": r1, "r2": r2})
    result = cbrkit.retrieval.apply_query({}, "a", retriever)

    assert len(result.casebase) == 2
    assert result.casebase[1] == "a"
    assert result.casebase[2] == "b"
