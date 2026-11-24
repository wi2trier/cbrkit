from collections.abc import Mapping
from typing import Any

import cbrkit


def adapt_int(case: int, query: int) -> int:
    if case > query:
        return case - query

    return case + query


def test_reuse_simple(car_query):
    case = {
        "price": 12000,
        "year": 2008,
        "manufacturer": "audi",
        "make": "a6",
        "miles": 150000,
    }

    reuse_func = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "price": adapt_int,
                "make": cbrkit.adapt.strings.RegexAdaptation(
                    case_pattern="a[0-9]", query_pattern="a[0-9]", replacement="a4"
                ),
                "manufacturer": cbrkit.adapt.strings.RegexAdaptation(
                    case_pattern="(audi)", query_pattern="audi", replacement=r"\1-\1"
                ),
                "miles": cbrkit.adapt.numbers.Aggregate(),
            }
        ),
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": cbrkit.sim.numbers.Linear(max_distance=100000),
            }
        ),
    )

    result = cbrkit.cycle.run_query({"case": case}, car_query, reuse_func)

    assert result.casebase["case"] == {
        "price": 2000,
        "year": 2008,
        "manufacturer": "audi-audi",
        "make": "a4",
        "miles": 125000,
    }


def custom_adapt(case: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    adapted = {
        "price": case["price"] - query["price"],
        "year": case["year"],
        "manufacturer": f"{case['manufacturer']}-{query['year']}",
        "make": case["make"],
        "miles": (case["miles"] + query["miles"]) // 2,
    }

    return adapted


def test_reuse_custom(car_query):
    case = {
        "price": 12000,
        "year": 2008,
        "manufacturer": "audi",
        "make": "a6",
        "miles": 150000,
    }

    reuse_func = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=custom_adapt,
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "price": cbrkit.sim.numbers.Linear(max_distance=100000),
                "year": cbrkit.sim.numbers.Linear(max_distance=50),
                "make": cbrkit.sim.strings.Levenshtein(),
                "miles": cbrkit.sim.numbers.Linear(max_distance=100000),
            }
        ),
    )

    result = cbrkit.cycle.run_query({"case": case}, car_query, reuse_func)

    assert result.casebase["case"] == {
        "price": 2000,
        "year": 2008,
        "manufacturer": "audi-2010",
        "make": "a6",
        "miles": 125000,
    }


def test_reuse_nested(cars_yaml_casebase):
    query = {
        "miles": 100000,
        "model": {
            "manufacturer": "audi",
            "make": "a4",
        },
    }
    full_casebase: Mapping[int, Any] = cars_yaml_casebase
    casebase = {key: full_casebase[key] for key in list(full_casebase.keys())[:10]}

    reuse_func = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "miles": cbrkit.adapt.numbers.Aggregate(),
                "model": cbrkit.adapt.AttributeValueAdaptation(
                    attributes={
                        "make": cbrkit.adapt.strings.RegexAdaptation(
                            case_pattern="v.*", query_pattern=".*", replacement="vclass"
                        ),
                        "manufacturer": cbrkit.adapt.strings.RegexAdaptation(
                            case_pattern="mercedes-benz",
                            query_pattern=".*",
                            replacement="mercedes",
                        ),
                    }
                ),
            }
        ),
        measure=cbrkit.sim.AttributeValue(
            attributes={
                "miles": cbrkit.sim.numbers.Linear(max_distance=100000),
                "model": cbrkit.sim.AttributeValue(
                    attributes={
                        "make": cbrkit.sim.strings.Levenshtein(),
                        "manufacturer": cbrkit.sim.strings.Levenshtein(),
                    }
                ),
            }
        ),
    )

    result = cbrkit.cycle.run_query(casebase, query, reuse_func)

    assert len(result.casebase) == len(casebase)
    assert result.casebase[0]["model"] == {
        "make": "vclass",
        "manufacturer": "mercedes",
    }


def test_reuse_does_not_mutate_casebase() -> None:
    """Adaptation must not write back into the casebase it was retrieved from."""
    casebase = {"a": {"price": 10, "model": {"make": "vw"}}}
    query = {"price": 30, "model": {"make": "audi"}}
    original = {"price": 10, "model": {"make": "vw"}}

    reuse_func = cbrkit.reuse.Reuser(
        limit=None,
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={
                "price": cbrkit.adapt.numbers.Aggregate(),
                "model": cbrkit.adapt.AttributeValueAdaptation(
                    attributes={"make": lambda case, query: query}
                ),
            }
        ),
        measure=lambda x, y: 1.0,
    )

    result = cbrkit.cycle.run_query(casebase, query, reuse_func)

    assert casebase["a"] == original
    assert casebase["a"]["model"] is not result.casebase["a"]["model"]
    assert result.casebase["a"] == {"price": 20.0, "model": {"make": "audi"}}
