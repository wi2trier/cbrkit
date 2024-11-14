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
        adaptation_func=cbrkit.adapt.attribute_value(
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
        similarity_func=cbrkit.sim.attribute_value(
            attributes={
                "price": cbrkit.sim.numbers.linear(max=100000),
                "year": cbrkit.sim.numbers.linear(max=50),
                "make": cbrkit.sim.strings.levenshtein(),
                "miles": cbrkit.sim.numbers.linear(max=100000),
            }
        ),
    )

    result = cbrkit.reuse.apply_single(case, query, reuse_func)

    assert result.case == {
        "price": 2000,
        "year": 2008,
        "manufacturer": "audi-audi",
        "make": "a4",
        "miles": 125000,
    }
