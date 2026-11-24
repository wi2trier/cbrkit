import asyncio

import pytest

import cbrkit
from cbrkit.errors import Conflict
from cbrkit.model import Add, Case, Changes, Remove, Update, replan
from cbrkit.retain import Retainer
from cbrkit.sim.numbers import Linear


def test_retain_after_the_other_phases(medium_casebase, simple_query, sim_func_simple):
    storage = cbrkit.indexable.InMemory(medium_casebase)
    result = cbrkit.cycle.run_query(
        storage,
        simple_query,
        cbrkit.retrieve.Retriever(measure=sim_func_simple, limit=2),
        cbrkit.reuse.Reuser(
            limit=None,
            adaptation=cbrkit.adapt.AttributeValueAdaptation(
                attributes={"price": cbrkit.adapt.numbers.Aggregate()}
            ),
            measure=sim_func_simple,
        ),
        cbrkit.revise.Reviser(measure=sim_func_simple),
        Retainer(storage=storage, measure=sim_func_simple),
    )

    assert [change.key for change in result.changes.items] == [3, 4]
    assert result.ranking == (3, 4)
    assert len(asyncio.run(storage.get_index())) == 5


def test_retain_merges_duplicates_and_is_replanned_on_conflicts():
    storage = cbrkit.indexable.InMemory({0: 1.0, 1: 5.0})
    retainer = Retainer(
        storage=storage,
        measure=Linear(max_distance=10),
        duplicate=0.99,
        merge=cbrkit.adapt.numbers.Aggregate(),
    )
    batches = [({"a": 1.05, "b": 1.0, "c": 9.0}, 1.0)]
    changes = retainer.plan(asyncio.run(storage.get_index()), batches)

    assert [(c.op, c.key) for c in changes.items] == [("update", 0), ("add", None)]
    assert isinstance(changes.items[0], Update)
    assert changes.items[0].value == pytest.approx(1.0125)

    asyncio.run(storage.patch_index(upsert={2: 9.0}))

    with pytest.raises(Conflict):
        asyncio.run(changes.apply(storage))

    [(cases, similarities)] = asyncio.run(retainer(batches))

    assert [(s.op, s.rule) for s in similarities.values()] == [
        ("update", "duplicate"),
        ("update", "duplicate"),
    ]
    assert dict(cases) == {0: pytest.approx(1.0125), 2: 9.0}
    assert dict(asyncio.run(storage.get_index())) == {
        0: pytest.approx(1.0125),
        1: 5.0,
        2: 9.0,
    }


def test_retain_split_cases_by_their_problems():
    storage = cbrkit.indexable.InMemory({0: Case(problem=1.0, solution="small")})
    result = cbrkit.cycle.run_query(
        storage,
        Case(problem=1.0),
        cbrkit.retrieve.Retriever(measure=Linear(max_distance=10), limit=1),
        cbrkit.reuse.Reuser(
            limit=None,
            adaptation=lambda case, query: Case(
                problem=case.problem, solution=f"{case.solution}!"
            ),
            measure=Linear(max_distance=10),
        ),
        Retainer(storage=storage, measure=Linear(max_distance=10), duplicate=1.0),
    )

    assert result.changes.items == []
    assert asyncio.run(storage.get_index()) == {0: Case(problem=1.0, solution="small")}


def test_maintenance_merges_duplicates_and_removes_by_competence():
    # 0, 1 and 2 solve each other, 3 is an outlier and 4 duplicates 0
    storage = cbrkit.indexable.InMemory({0: 1.0, 1: 1.3, 2: 1.6, 3: 9.0, 4: 1.01})
    maintainer = cbrkit.maintain.Maintainer(
        measure=Linear(max_distance=10),
        duplicate=0.99,
        merge=cbrkit.adapt.numbers.Aggregate(),
        competence=0.9,
        remove=frozenset({"auxiliary", "support", "spanning"}),
    )

    changes = maintainer.plan(asyncio.run(storage.get_index()), uses={1: 5})

    assert [(c.op, c.key, c.rule) for c in changes.items] == [
        ("update", 0, "duplicate"),
        ("remove", 4, "duplicate"),
        ("remove", 2, "competence"),
        ("remove", 1, "competence"),
    ]
    assert {a.key: a.category for a in changes.reviews}[3] == "pivotal"

    applied = asyncio.run(replan(maintainer, storage, uses={1: 5}))

    assert applied.applied
    assert dict(asyncio.run(storage.get_index())) == {0: 1.005, 3: 9.0}


def test_maintenance_removes_inconsistent_cases():
    # problems are the first, solutions the second value, case 1 has a deviating solution
    cases = {0: (1.0, 1.0), 1: (1.1, 9.0), 2: (1.2, 1.2)}

    def problem(case: tuple[float, float], query: tuple[float, float]) -> float:
        return Linear(max_distance=10)(case[0], query[0])

    def solution(case: tuple[float, float], query: tuple[float, float]) -> float:
        return Linear(max_distance=10)(case[1], query[1])

    changes = cbrkit.maintain.Maintainer(
        measure=problem, competence=0.9, solution_measure=solution, consistency=0.5
    ).plan(cases)

    assert [(c.op, c.key, c.rule) for c in changes.items] == [
        ("remove", 1, "consistency")
    ]


def test_competence_categories_follow_smyth_and_keane():
    # 0 covers 1 and 2, which only cover themselves and 0, so they are auxiliary,
    # and 0 supports them since they solve it and their coverage is part of its own
    reviews = (
        cbrkit.maintain.Maintainer(measure=Linear(max_distance=10), competence=0.85)
        .plan({0: 5.0, 1: 4.0, 2: 6.0, 3: 20.0})
        .reviews
    )

    assert [a.category for a in reviews] == [
        "support",
        "auxiliary",
        "auxiliary",
        "pivotal",
    ]


@pytest.mark.parametrize(
    "second",
    [
        Update(key=0, value="b", rule="duplicate", reason=""),
        Remove(key=0, rule="competence", reason=""),
    ],
)
def test_changes_refuse_changing_a_case_twice(second):
    changes = Changes[int, str](
        items=[Update(key=0, value="a", rule="duplicate", reason=""), second]
    )

    with pytest.raises(ValueError, match="twice"):
        asyncio.run(changes.apply(cbrkit.indexable.InMemory({0: "x"})))


def test_keyed_additions_insert_or_conflict():
    storage = cbrkit.indexable.InMemory({0: "x"})
    changes = Changes[int, str](
        items=[Add(key=5, value="y", reason=""), Add(value="z", reason="")]
    )

    assert [change.key for change in asyncio.run(changes.apply(storage)).items] == [
        5,
        6,
    ]

    with pytest.raises(Conflict):
        asyncio.run(
            Changes[int, str](items=[Add(key=0, value="y", reason="")]).apply(storage)
        )
