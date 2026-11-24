import asyncio
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pytest

import cbrkit

from .conftest import SearchingRetriever, _custom_numeric_sim


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
def test_retrieve_process_pool(cars_csv_casebase):
    query_name = 42
    casebase = cars_csv_casebase
    executor = ProcessPoolExecutor(2)
    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                    taxonomy=cbrkit.sim.taxonomy.load("./data/cars-taxonomy.yaml"),
                    measure=cbrkit.sim.taxonomy.WuPalmer(),
                ),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": _custom_numeric_sim,
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        executor=executor,
        chunksize=5,
    )

    with executor:
        result = cbrkit.cycle.run_query(casebase, casebase[query_name], retriever)

    assert len(result.ranking) == 999


def test_retrieve_dataframe(cars_csv_casebase):
    query_name = 42
    casebase = cars_csv_casebase
    query = casebase[query_name]
    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                    taxonomy=cbrkit.sim.taxonomy.load("./data/cars-taxonomy.yaml"),
                    measure=cbrkit.sim.taxonomy.WuPalmer(),
                ),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": cbrkit.sim.numbers.Linear(max_distance=1000000),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )
    result = cbrkit.cycle.run_query(
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

    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                    taxonomy=cbrkit.sim.taxonomy.load("./data/cars-taxonomy.yaml"),
                    measure=cbrkit.sim.taxonomy.WuPalmer(),
                ),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": _custom_numeric_sim,
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        limit=5,
    )
    result = cbrkit.cycle.run_query(
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
    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "model": cbrkit.sim.AttributeValue(
                    attributes={
                        "make": cbrkit.sim.strings.Levenshtein(),
                        "manufacturer": cbrkit.sim.taxonomy.Taxonomy(
                            taxonomy=cbrkit.sim.taxonomy.load(
                                "./data/cars-taxonomy.yaml"
                            ),
                            measure=cbrkit.sim.taxonomy.WuPalmer(),
                        ),
                    }
                ),
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        min_similarity=0.5,
    )
    result = cbrkit.cycle.run_query(
        casebase,
        query,
        retriever,
    )

    assert len(casebase) == 999
    assert result.similarities[query_name].value == 1.0
    assert result.ranking[0] == query_name

    model_sim = result.similarities[query_name].similarities["model"]

    assert isinstance(model_sim, cbrkit.model.AttributeValueSim)
    assert model_sim.value == 1.0
    assert model_sim.similarities["make"] == 1.0


def test_retrieve_interpolation_groups() -> None:
    casebase = cbrkit.retrieve.as_groups({"low": 0.0, "middle": 10.0})
    retriever = cbrkit.retrieve.Group(
        retriever=cbrkit.retrieve.Retriever(
            measure=interpolation_relation, min_similarity=1.0
        ),
        size=2,
    )

    result = cbrkit.cycle.run_query(casebase, (4.0,), retriever)

    assert result.ranking == (("low", "middle"),)
    assert result.casebase[("low", "middle")] == (0.0, 10.0)


def test_retrieve_extrapolation_groups() -> None:
    casebase = cbrkit.retrieve.as_groups({"first": 0.0, "second": 2.0, "third": 5.0})
    retriever = cbrkit.retrieve.Group(
        retriever=cbrkit.retrieve.Retriever(
            measure=extrapolation_relation, min_similarity=1.0
        ),
        size=3,
        ordered=True,
    )

    result = cbrkit.cycle.run_query(casebase, (7.0,), retriever)

    assert ("first", "second", "third") in result.ranking
    assert result.similarities[("first", "second", "third")] == 1.0


def test_grouped_similarity_wrappers() -> None:
    metric = cbrkit.sim.numbers.Linear(max_distance=10)
    source_proximity = cbrkit.sim.collections.IsolatedMapping(element_measure=metric)
    prediction_quality = cbrkit.sim.transpose_singleton(metric)

    assert float(source_proximity((0.0, 10.0), (4.0,))) == 0.6
    assert prediction_quality([((4.0,), (5.0,))]) == [0.9]


def test_group_rejects_invalid_size() -> None:
    retriever = cbrkit.retrieve.Group(
        retriever=cbrkit.retrieve.Retriever(measure=lambda x, y: 1.0), size=0
    )

    with pytest.raises(ValueError, match="size must be at least 1"):
        asyncio.run(retriever([(cbrkit.retrieve.as_groups({"case": 1}), (1,))]))


def test_retrieve_groups_mac_fac(cars_csv_casebase, sim_func_simple) -> None:
    """A cheap retriever pre-filters the source cases before the expansion."""
    query = cars_csv_casebase[42]
    singletons = cbrkit.retrieve.as_groups(cars_csv_casebase)

    mac = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.transpose_singleton(sim_func_simple), limit=20
    )
    fac = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": _bracket_width(100000),
                "year": _bracket_width(50),
            },
            value_getter=cbrkit.helpers.broadcast_getter,
        ),
        limit=3,
        min_similarity=0.001,
    )

    steps = cbrkit.cycle.run(
        singletons, {"q": (query,)}, mac, cbrkit.retrieve.Group(retriever=fac, size=2)
    ).steps
    first, result = steps[0]["q"], steps[1]["q"]

    assert len(steps) == 2
    assert len(first.ranking) == 20
    assert all(len(key) == 1 for key in first.ranking)
    assert all(len(key) == 2 for key in result.ranking)

    for key in result.ranking:
        lower, upper = (case["price"] for case in result.casebase[key])
        assert min(lower, upper) <= query["price"] <= max(lower, upper)


def test_group_expansion() -> None:
    """Groups are enumerated over the source keys and their cases concatenated."""

    def narrow(group: tuple[float, ...]) -> bool:
        return max(group) - min(group) <= 15.0

    casebase = {"a": 0.0, "b": 10.0, "c": 25.0}
    retriever = cbrkit.retrieve.Group(
        retriever=lambda batches: [(cb, {}) for cb, _ in batches],
        size=2,
        filter=narrow,
    )

    results = asyncio.run(retriever([(cbrkit.retrieve.as_groups(casebase), (4.0,))]))

    assert results[0][0] == {("a", "b"): (0.0, 10.0), ("b", "c"): (10.0, 25.0)}


def test_run_reads_the_current_cases_of_a_storage() -> None:
    storage = cbrkit.indexable.InMemory({0: "a", 1: "b", 2: "c"})
    retriever = cbrkit.retrieve.Retriever(measure=cbrkit.sim.generic.Equality())

    result = cbrkit.cycle.run_query(storage, "a", retriever)
    assert len(result.casebase) == 3
    assert result.similarities[0] == 1.0

    asyncio.run(storage.patch_index(upsert={3: "d"}, delete=[1, 2]))
    result = cbrkit.cycle.run_query(storage, "a", retriever)
    assert set(result.casebase) == {0, 3}

    assert cbrkit.cycle.run_query({}, "a", retriever).casebase == {}


def test_combine_retrievers_that_search_their_storages() -> None:
    first = SearchingRetriever({1: "a", 2: "b"})
    second = SearchingRetriever({2: "b", 3: "c"})

    result = cbrkit.cycle.run_query(
        None, "a", cbrkit.retrieve.Combine(retrievers=[first, second], limit=2)
    )

    assert set(result.casebase) == {1, 2}
    assert result.casebase[1] == "a"


def test_fuse_runs_synchronous_and_asynchronous_retrievers() -> None:
    async def searching(
        batches: cbrkit.typing.Batches[int, str], /
    ) -> cbrkit.typing.Results[int, str]:
        return [({2: "b", 3: "c"}, {2: 1.0, 3: 0.5}) for _ in batches]

    fused = cbrkit.retrieve.Fuse(
        retrievers=[SearchingRetriever({1: "a", 2: "b"}), searching], k=0, limit=1
    )

    assert cbrkit.cycle.run_query(None, "a", fused).similarities == {2: 1.5}


def test_fuse_queries():
    retriever = cbrkit.retrieve.Retriever(
        measure=cbrkit.sim.numbers.Linear(max_distance=10)
    )
    result = cbrkit.cycle.run(
        {"a": 0, "b": 5, "c": 10}, {"low": 0, "high": 10}, retriever
    )
    fused = cbrkit.retrieve.fuse([r.hits() for r in result.values()], limit=2)

    assert [hit.key for hit in fused] == ["a", "c"]
    assert [hit.score for hit in fused] == [1 / 61 + 1 / 63] * 2


def test_combine_intersection():
    def first(x, y):
        return 1.0 if x == y else 0.0

    combined = cbrkit.retrieve.Combine(
        retrievers=[
            cbrkit.retrieve.Retriever(measure=first, limit=2),
            cbrkit.retrieve.Retriever(measure=first, min_similarity=1),
        ],
        default=None,
    )
    result = cbrkit.cycle.run_query({"a": 1, "b": 2, "c": 3}, 1, combined)

    assert result.similarities == {"a": 1.0}


def test_result_json_keeps_detailed_similarities() -> None:
    similarity = cbrkit.sim.AttributeValue(
        attributes={"a": cbrkit.sim.generic.Equality()}
    )
    result = cbrkit.cycle.run_query(
        {"x": {"a": 1}, "y": {"a": 2}},
        {"a": 1},
        cbrkit.retrieve.Retriever(measure=similarity),
    )
    loaded = type(result).model_validate_json(result.model_dump_json())

    assert loaded.similarities == result.similarities
    assert isinstance(loaded.similarities["x"], cbrkit.model.AttributeValueSim)
    assert float(loaded.similarities["x"]) == 1.0
