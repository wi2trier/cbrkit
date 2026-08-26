from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import cbrkit

from .conftest import FakeIndexableRetriever, _custom_numeric_sim


def interpolation_relation(case: tuple[float, ...], query: tuple[float, ...]) -> float:
    target = cbrkit.helpers.singleton(query)
    lower, upper = case

    return float(min(lower, upper) <= target <= max(lower, upper))


def _bracket_width(span: float):
    """Local measure: how tightly the pair `x` brackets the singleton query `y`."""

    def relation(x: tuple[float, ...], y: tuple[float, ...]) -> float:
        lower, upper = sorted(x)
        target = cbrkit.helpers.singleton(y)

        if not lower <= target <= upper:
            return 0.0

        return max(0.0, 1.0 - (upper - lower) / span)

    return relation


def extrapolation_relation(case: tuple[float, ...], query: tuple[float, ...]) -> float:
    target = cbrkit.helpers.singleton(query)
    first, second, third = case

    return 1.0 / (1.0 + abs((second - first) - (target - third)))


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


def test_retrieve_interpolation_groups() -> None:
    casebase = cbrkit.retrieval.as_groups({"low": 0.0, "middle": 10.0})
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.group(cbrkit.retrieval.build(interpolation_relation), size=2),
        min_similarity=1.0,
    )

    result = cbrkit.retrieval.apply_query(casebase, (4.0,), retriever)

    assert result.ranking == (("low", "middle"),)
    assert result.casebase[("low", "middle")] == (0.0, 10.0)


def test_retrieve_extrapolation_groups() -> None:
    casebase = cbrkit.retrieval.as_groups({"first": 0.0, "second": 2.0, "third": 5.0})
    retriever = cbrkit.retrieval.dropout(
        cbrkit.retrieval.group(
            cbrkit.retrieval.build(extrapolation_relation), size=3, ordered=True
        ),
        min_similarity=1.0,
    )

    result = cbrkit.retrieval.apply_query(casebase, (7.0,), retriever)

    assert ("first", "second", "third") in result.ranking
    assert result.similarities[("first", "second", "third")] == 1.0


def test_grouped_similarity_wrappers() -> None:
    metric = cbrkit.sim.numbers.linear(max=10)
    source_proximity = cbrkit.sim.collections.isolated_mapping(metric)
    prediction_quality = cbrkit.sim.transpose_singleton(metric)

    assert source_proximity((0.0, 10.0), (4.0,)) == 0.6
    assert prediction_quality([((4.0,), (5.0,))]) == [0.9]


def test_group_rejects_invalid_size() -> None:
    retriever = cbrkit.retrieval.group(cbrkit.retrieval.build(lambda x, y: 1.0), size=0)

    with pytest.raises(ValueError, match="size must be at least 1"):
        retriever([(cbrkit.retrieval.as_groups({"case": 1}), (1,))])


def test_retrieve_groups_mac_fac(cars_csv_casebase, sim_func_simple) -> None:
    """A cheap retriever pre-filters the source cases before the expansion."""
    query = cars_csv_casebase[42]
    singletons = cbrkit.retrieval.as_groups(cars_csv_casebase)

    mac = cbrkit.retrieval.build(cbrkit.sim.transpose_singleton(sim_func_simple))
    fac = cbrkit.retrieval.build(
        cbrkit.sim.attribute_value(
            attributes={
                "price": _bracket_width(100000),
                "year": _bracket_width(50),
            },
            value_getter=cbrkit.helpers.broadcast_getter,
        )
    )

    result = cbrkit.retrieval.apply_query(
        singletons,
        (query,),
        [
            cbrkit.retrieval.dropout(mac, limit=20),
            cbrkit.retrieval.dropout(
                cbrkit.retrieval.group(fac, size=2),
                limit=3,
                min_similarity=0.001,
            ),
        ],
    )

    assert len(result.steps) == 2
    assert len(result.first_step.ranking) == 20
    assert all(len(key) == 1 for key in result.first_step.ranking)
    assert all(len(key) == 2 for key in result.ranking)

    for key in result.ranking:
        lower, upper = (case["price"] for case in result.casebase[key])
        assert min(lower, upper) <= query["price"] <= max(lower, upper)


def test_group_expansion() -> None:
    """Groups are enumerated over the source keys and their cases concatenated."""

    def narrow(group: tuple[float, ...]) -> bool:
        return max(group) - min(group) <= 15.0

    casebase = {"a": 0.0, "b": 10.0, "c": 25.0}
    retriever = cbrkit.retrieval.group(
        lambda batches: [(cb, {}) for cb, _ in batches],
        size=2,
        filter_func=narrow,
    )

    results = retriever([(cbrkit.retrieval.as_groups(casebase), (4.0,))])

    assert results[0][0] == {("a", "b"): (0.0, 10.0), ("b", "c"): (10.0, 25.0)}


def test_retrieve_indexed_lifecycle() -> None:
    retriever = FakeIndexableRetriever()

    with pytest.raises(ValueError, match="Call put_index\\(\\) first"):
        cbrkit.retrieval.apply_query({}, "a", retriever)

    retriever.upsert_index({1: "a", 2: "b"})
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 2

    result = cbrkit.retrieval.apply_query_indexed("a", retriever)
    assert len(result.casebase) == 2

    retriever.upsert_index({3: "c"})
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 3

    retriever.delete_index([2])
    result = cbrkit.retrieval.apply_query({}, "a", retriever)
    assert len(result.casebase) == 2
    assert 2 not in result.casebase


def test_retrieve_indexed_combine() -> None:
    r1 = FakeIndexableRetriever()
    r1.put_index({1: "a", 2: "b"})

    r2 = FakeIndexableRetriever()
    r2.put_index({1: "a", 2: "b"})

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

    retriever.upsert_index({3: "d"})
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
    retriever.upsert_index({"a": "x", "b": "y"})
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
