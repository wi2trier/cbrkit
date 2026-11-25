"""The CBR system built from a specification, without any network."""

import asyncio
from collections.abc import Sequence

import pytest
from pydantic import TypeAdapter

import cbrkit
from cbrkit import spec
from cbrkit.system import System
from cbrkit.system.operations import operations

CASES = {
    0: {"price": 10.0, "fuel": "petrol"},
    1: {"price": 20.0, "fuel": "diesel"},
    2: {"price": 40.0, "fuel": "petrol"},
}


def config(**sections) -> spec.Config:
    return spec.infer(CASES, cbrkit.io.Source(format="csv")).model_copy(update=sections)


def test_retrieve_reuse_and_revise():
    system = System(
        config=config(
            reuse=cbrkit.reuse.Reuser(
                adaptation=cbrkit.adapt.AttributeValueAdaptation(
                    attributes={"price": cbrkit.adapt.generic.Null(target="query")}
                )
            ),
            revise=cbrkit.revise.Reviser(measure=config().measure, min_similarity=0.9),
        ),
        storage=cbrkit.indexable.InMemory(CASES),
    )
    query = {"price": 18.0, "fuel": "diesel"}

    retrieved = asyncio.run(system.retrieve(query, limit=2))
    reused = asyncio.run(system.reuse(query, limit=1))
    revised = asyncio.run(system.revise(query, [CASES[1], CASES[2]]))
    explained = asyncio.run(system.retrieve(query, limit=1, explain=True))
    keyed = asyncio.run(system.reuse(key=1, limit=1, explain=True))

    assert [hit.key for hit in retrieved.hits] == [1, 0]
    assert reused.hits[0].value["price"] == 18.0
    assert keyed.hits[0].key == 0 and keyed.hits[0].explanation is not None
    assert [hit.key for hit in revised.hits] == [0]
    explanation = explained.hits[0].explanation
    assert explanation is not None and set(explanation.attributes) == {"price", "fuel"}


def test_outcomes_dump_graphs_and_frozendicts_as_json():
    graph = cbrkit.model.graph.from_dict({"nodes": {"a": {"name": "x"}}, "edges": {}})
    cases = {"g": graph}
    system = System(
        config=spec.infer(cases, cbrkit.io.Source(format="json", graphs=True)),
        storage=cbrkit.indexable.InMemory(cases),
    )
    outcome = asyncio.run(system.retrieve(graph, explain=True))
    hit = outcome.model_dump(mode="json")["hits"][0]
    record = cbrkit.model.Hit(
        key=0, value={"a": cbrkit.helpers.frozendict(b=1)}, score=1.0
    )

    assert hit["value"] == {
        "version": 1,
        "nodes": {"a": {"name": "x"}},
        "edges": {},
        "value": None,
    }
    assert record.model_dump(mode="json")["value"] == {"a": {"b": 1}}


def test_unbound_names_fail_on_construction():
    judge = cbrkit.sim.llm.Critique(
        model=cbrkit.sim.llm.LanguageModel("judge"), instructions="Judge."
    )

    with pytest.raises(cbrkit.errors.Unbound, match="LanguageModel 'judge'"):
        System(
            config=config(revise=cbrkit.revise.Reviser(measure=judge)),
            storage=cbrkit.indexable.InMemory(CASES),
        )


def test_load_binds_the_functions_of_the_host(tmp_path):
    path = tmp_path / "cases.csv"
    judge = cbrkit.sim.llm.Critique(
        model=cbrkit.sim.llm.LanguageModel("judge"), instructions="Judge."
    )

    async def host[T](batch: Sequence[str], response_type: type[T], /) -> Sequence[T]:
        return []

    cbrkit.io.save(path, cbrkit.io.Source(format="csv"), CASES)
    spec.write(
        cbrkit.io.companion(path, "sidecar"),
        config(revise=cbrkit.revise.Reviser(measure=judge)),
    )
    system = System.load(path, models={"judge": host})

    assert system.models == {"judge": host}


def test_retain_and_maintain(tmp_path):
    path = tmp_path / "cases.csv"
    cbrkit.io.save(path, cbrkit.io.Source(format="csv"), CASES)
    spec.write(cbrkit.io.companion(path, "sidecar"), config())

    spec.write(cbrkit.io.companion(path, "sidecar"), config(retain=None))
    off = asyncio.run(
        System.load(path).retain({"price": 15.0}, [{"price": 15.0, "fuel": "gas"}])
    )

    assert off.changes is not None and not off.changes.items

    spec.write(
        cbrkit.io.companion(path, "sidecar"),
        config(
            retain=cbrkit.retain.Retainer(),
            maintain=cbrkit.maintain.Maintainer(competence=0.9),
        ),
    )
    system = System.load(path)
    solutions = [{"price": 15.0, "fuel": "gas"}, {"price": 400.0, "fuel": "gas"}]
    dry = asyncio.run(system.retain({"price": 15.0}, solutions, dry_run=True))
    stored = asyncio.run(system.retain({"price": 15.0}, solutions))

    assert dry.changes is not None and not dry.changes.applied and not dry.hits
    assert [(c.op, c.rule) for c in dry.changes.items] == [("add", "solution")] * 2
    assert stored.changes is not None and stored.changes.applied
    assert [c.key for c in stored.changes.items] == [3, 4]
    assert asyncio.run(System.load(path).storage.get_index())[3] == {
        "price": 15.0,
        "fuel": "gas",
    }

    maintained = asyncio.run(system.maintain(dry_run=True))
    assert maintained.changes is not None and len(maintained.changes.reviews) == 5


def test_language_models_of_the_providers_adapt_and_critique():
    prompts: list[str] = []

    async def model[T](batch: Sequence[str], response_type: type[T], /) -> Sequence[T]:
        prompts.extend(batch)
        response = (
            {"score": 0.8, "issues": ["used"]}
            if response_type is cbrkit.sim.llm.Judgement
            else {"price": 19.0, "fuel": "gas"}
        )

        return [TypeAdapter(response_type).validate_python(response) for _ in batch]

    system = System(
        config=config(
            reuse=cbrkit.reuse.Reuser(
                adaptation=cbrkit.adapt.llm.Rewrite(
                    model=cbrkit.sim.llm.LanguageModel("writer"),
                    instructions="Match the price.",
                    attributes=["price"],
                )
            ),
            revise=cbrkit.revise.Reviser(
                measure=cbrkit.sim.llm.Critique(
                    model=cbrkit.sim.llm.LanguageModel("judge"),
                    instructions="Judge the offer.",
                )
            ),
            providers=spec.Providers(models={"judge": "test:judge"}),
        ),
        storage=cbrkit.indexable.InMemory(CASES),
        models={"writer": model, "judge": model},
    )
    outcome = asyncio.run(system.cycle({"price": 18.0, "fuel": "diesel"}, limit=1))
    hit = outcome.hits[0]

    assert hit.value == {"price": 19.0, "fuel": "diesel"}
    assert hit.score == cbrkit.model.CritiqueSim(
        value=0.8, issues=("used",), model="judge"
    )
    assert prompts[0].startswith("Match the price.")


def test_providers_bind_pydantic_ai_unless_the_host_passes_a_function():
    pytest.importorskip("pydantic_ai")
    rewrite = cbrkit.adapt.llm.Rewrite(
        model=cbrkit.sim.llm.LanguageModel("a"), instructions="Rewrite."
    )
    critique = cbrkit.sim.llm.Critique(
        model=cbrkit.sim.llm.LanguageModel("b"), instructions="Judge."
    )
    unbound = config(
        reuse=cbrkit.reuse.Reuser(adaptation=rewrite),
        revise=cbrkit.revise.Reviser(measure=critique),
        providers=spec.Providers(models={"a": "test:a", "b": "test:b"}),
    )
    host = cbrkit.sim.llm.PydanticAI("test")
    bound = spec.bind(unbound, *spec.providers(unbound, models={"b": host}))

    with pytest.raises(cbrkit.errors.Unbound):
        spec.bind(unbound, models={"b": host})

    assert bound.reuse is not None and bound.revise is not None
    rewrite, critique = bound.reuse.adaptation, bound.revise.measure
    assert isinstance(rewrite, cbrkit.adapt.llm.Rewrite)
    assert isinstance(critique, cbrkit.sim.llm.Critique)
    assert rewrite.model.func == cbrkit.sim.llm.PydanticAI("test:a")
    assert critique.model.func is host


def test_operations_are_pydantic_ai_tools():
    toolset = pytest.importorskip("pydantic_ai").FunctionToolset(
        operations(System(config=config(), storage=cbrkit.indexable.InMemory(CASES)))
    )
    schema = toolset.tools["retrieve"].function_schema.json_schema

    assert set(toolset.tools) == {"retrieve", "reuse", "retain", "cycle"}
    assert schema["properties"]["limit"]["anyOf"][0]["maximum"] == 100


def test_graph_queries_stay_within_the_node_pair_budget():
    graphs = {
        key: cbrkit.model.graph.from_dict({"nodes": {"x": {"v": 1.0}, "y": {"v": 2.0}}})
        for key in ("a", "b")
    }
    config = spec.infer(graphs, cbrkit.io.Source(format="json", graphs=True))
    system = System(
        config=config.model_copy(update={"limits": spec.Limits(max_node_pairs=6)}),
        storage=cbrkit.indexable.InMemory(graphs),
    )

    assert [hit.key for hit in asyncio.run(system.retrieve(key="a")).hits] == ["b"]

    with pytest.raises(cbrkit.errors.Refused, match="node pairs"):
        asyncio.run(system.retrieve(graphs["a"]))

    with pytest.raises(cbrkit.errors.Refused, match="node pairs"):
        asyncio.run(system.cycle(graphs["a"]))


@pytest.mark.parametrize("reuse", [False, True])
def test_the_cycle_retains_what_reuse_proposes(reuse: bool):
    reuser = cbrkit.reuse.Reuser(
        adaptation=cbrkit.adapt.AttributeValueAdaptation(
            attributes={"price": cbrkit.adapt.generic.Null(target="query")}
        )
    )
    storage = cbrkit.indexable.InMemory(CASES)
    system = System(
        config=config(retain=cbrkit.retain.Retainer(), reuse=reuser if reuse else None),
        storage=storage,
    )
    cycle = next(op for op in operations(system) if op.__name__ == "cycle")

    outcome = asyncio.run(
        system.cycle({"price": 18.0, "fuel": "diesel"}, limit=2, explain=True)
    )
    stored = asyncio.run(storage.get_index())
    explanation = outcome.hits[0].explanation

    assert cycle.read_only is not reuse
    # Reuse proposes one solution, the best adapted case.
    assert [hit.key for hit in outcome.hits] == ([1] if reuse else [1, 0])
    assert explanation is not None and set(explanation.attributes) == {"price", "fuel"}
    assert (outcome.changes is not None) is reuse
    assert len(stored) == len(CASES) + reuse
    assert not reuse or stored[3] == {"price": 18.0, "fuel": "diesel"}


def test_the_cycle_retains_no_case_that_reuse_left_unchanged():
    storage = cbrkit.indexable.InMemory(CASES)
    system = System(config=config(), storage=storage)

    outcome = asyncio.run(system.cycle({"price": 18.0, "fuel": "diesel"}))

    assert [hit.key for hit in outcome.hits] == [1]
    assert outcome.changes is not None and not outcome.changes.items
    assert asyncio.run(storage.get_index()) == CASES


def test_revise_needs_its_own_measure():
    unchecked = config(revise=cbrkit.revise.Reviser(min_similarity=0.5))

    with pytest.raises(ValueError, match="quality of solutions"):
        spec.Config.model_validate(unchecked.model_dump(mode="json"))
