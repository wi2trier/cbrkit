from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import cbrkit

from .conftest import FakeIndexableRetriever, _custom_numeric_sim


@pytest.mark.skip(reason="this test is slow on macOS")
def test_retrieve_multiprocessing(cars_csv_casebase):
    query_name = 42
    casebase = cars_csv_casebase
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


def test_retrieve_dataframe(cars_csv_casebase):
    query_name = 42
    casebase = cars_csv_casebase
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


def test_retrieve_dataframe_custom_query(cars_csv_casebase, car_query):
    casebase = cars_csv_casebase

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
        car_query,
        retriever,
    )

    assert len(result.similarities) == 5
    assert len(result.ranking) == 5
    assert len(result.casebase) == 5


def test_retrieve_nested(cars_yaml_casebase):
    query_name = 42
    casebase: Mapping[int, Any] = cars_yaml_casebase
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


def test_retrieve_indexed_lifecycle() -> None:
    retriever = FakeIndexableRetriever()

    # query before any index raises
    with pytest.raises(ValueError, match="Call create_index\\(\\) first"):
        cbrkit.retrieval.apply_query({}, "a", retriever)

    # update on empty index delegates to create
    retriever.update_index({1: "a", 2: "b"})
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 2

    # also works via apply_query_indexed
    result = cbrkit.retrieval.apply_query_indexed("a", retriever)
    assert len(result.casebase) == 2

    # update
    retriever.update_index({3: "c"})
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 3

    # delete
    retriever.delete_index([2])
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 2
    assert 2 not in result.casebase


def test_retrieve_indexed_combine() -> None:
    r1 = FakeIndexableRetriever()
    r1.create_index({1: "a", 2: "b"})

    r2 = FakeIndexableRetriever()
    r2.create_index({1: "a", 2: "b"})

    for retriever in [
        cbrkit.retrieval.combine([r1, r2]),
        cbrkit.retrieval.combine({"r1": r1, "r2": r2}),
    ]:
        result = cbrkit.retrieval.apply_query({}, "a", retriever)
        assert len(result.casebase) == 2
        assert result.casebase[1] == "a"


def test_retrieve_persist() -> None:
    cb: dict[int, str] = {0: "a", 1: "b", 2: "c"}
    retriever = cbrkit.retrieval.persist(
        retriever_func=cbrkit.retrieval.build(cbrkit.sim.generic.equality()),
        casebase=cb,
    )

    # indexed retrieval and CRUD
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 3
    assert result.similarities[0] == 1.0

    retriever.update_index({3: "d"})
    retriever.delete_index([1, 2])
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 2
    assert 3 in result.casebase


def test_retrieve_persist_file(tmp_path: Path) -> None:
    json_path = tmp_path / "casebase.json"

    # mutations auto-persist to disk
    retriever = cbrkit.retrieval.persist(
        retriever_func=cbrkit.retrieval.build(cbrkit.sim.generic.equality()),
        path=json_path,
    )
    retriever.update_index({"a": "x", "b": "y"})
    assert json_path.exists()

    # new instance loads persisted data
    retriever2 = cbrkit.retrieval.persist(
        retriever_func=cbrkit.retrieval.build(cbrkit.sim.generic.equality()),
        path=json_path,
    )
    assert dict(retriever2.index) == {"a": "x", "b": "y"}

    # retain integration: index mutations auto-persist
    storage = cbrkit.retain.indexable(
        key_func=lambda keys: chr(ord(max(keys, default="`")) + 1),
        indexable_func=retriever2,
    )
    retainer = cbrkit.retain.build(
        assess_func=cbrkit.sim.generic.equality(),
        storage_func=storage,
    )
    retainer([(dict(retriever2.index), "test")])
    assert len(retriever2.index) == 4

    # verify persisted by loading fresh
    retriever3 = cbrkit.retrieval.persist(
        retriever_func=cbrkit.retrieval.build(cbrkit.sim.generic.equality()),
        path=json_path,
    )
    assert len(retriever3.index) == 4
