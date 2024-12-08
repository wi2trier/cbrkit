from collections.abc import Mapping
from typing import Any

import cbrkit


def adapt_int(case: int, query: int) -> int:
    if case > query:
        return case - query

    return case + query


def test_reuse_simple():
    query = {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }
    case = {
        "price": 12000,
        "year": 2008,
        "manufacturer": "audi",
        "make": "a6",
        "miles": 150000,
    }

    reuse_func = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "price": adapt_int,
                "make": cbrkit.adapt.strings.regex("a[0-9]", "a[0-9]", "a4"),
                "manufacturer": cbrkit.adapt.strings.regex(
                    "audi",
                    "audi",
                    lambda case, query: f"{case.string}-{query.string}",
                ),
                "miles": cbrkit.adapt.numbers.aggregate("mean"),
            }
        ),
        retriever_func=cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": cbrkit.sim.numbers.linear(max=100000),
                }
            )
        ),
    )

    result = cbrkit.reuse.apply_pair(case, query, reuse_func)

    assert result.case == {
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


def test_reuse_custom():
    query = {
        "price": 10000,
        "year": 2010,
        "manufacturer": "audi",
        "make": "a4",
        "miles": 100000,
    }
    case = {
        "price": 12000,
        "year": 2008,
        "manufacturer": "audi",
        "make": "a6",
        "miles": 150000,
    }

    reuse_func = cbrkit.reuse.build(
        custom_adapt,
        retriever_func=cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "price": cbrkit.sim.numbers.linear(max=100000),
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": cbrkit.sim.numbers.linear(max=100000),
                }
            )
        ),
    )

    result = cbrkit.reuse.apply_pair(case, query, reuse_func)

    assert result.case == {
        "price": 2000,
        "year": 2008,
        "manufacturer": "audi-2010",
        "make": "a6",
        "miles": 125000,
    }


def test_reuse_nested():
    query = {
        "miles": 100000,
        "model": {
            "manufacturer": "audi",
            "make": "a4",
        },
    }
    full_casebase: Mapping[int, Any] = cbrkit.loaders.path("data/cars-1k.yaml")
    casebase = {key: full_casebase[key] for key in list(full_casebase.keys())[:10]}

    reuse_func = cbrkit.reuse.build(
        cbrkit.adapt.attribute_value(
            attributes={
                "miles": cbrkit.adapt.numbers.aggregate("mean"),
                "model": cbrkit.adapt.attribute_value(
                    {
                        "make": cbrkit.adapt.strings.regex("v.*", ".*", "vclass"),
                        "manufacturer": cbrkit.adapt.strings.regex(
                            "mercedes-benz", ".*", "mercedes"
                        ),
                    }
                ),
            }
        ),
        retriever_func=cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "miles": cbrkit.sim.numbers.linear(max=100000),
                    "model": cbrkit.sim.attribute_value(
                        attributes={
                            "make": cbrkit.sim.strings.levenshtein(),
                            "manufacturer": cbrkit.sim.strings.levenshtein(),
                        }
                    ),
                }
            )
        ),
    )

    result = cbrkit.reuse.apply_query(casebase, query, reuse_func)

    assert len(result.casebase) == len(casebase)
    assert result.casebase[0]["model"] == {
        "make": "vclass",
        "manufacturer": "mercedes",
    }
