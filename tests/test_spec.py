"""Declarative similarity specifications: inference, merging, sidecars, refinement and explanations."""

import asyncio
import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import numpy as np
import orjson
import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError

import cbrkit
from cbrkit import spec
from cbrkit.schema import Family, extend, serializable
from cbrkit.sim import AttributeValue, Measure
from cbrkit.sim.embed import Embedding
from cbrkit.typing import NumpyArray

MEASURE = TypeAdapter[Measure[Any, Any]](Measure[Any, Any])


def bpmn(tasks: list[tuple[str, str]]) -> str:
    nodes = "".join(
        f'<{kind} id="t{i}" name="{name}"/>' for i, (kind, name) in enumerate(tasks)
    )
    flows = "".join(
        f'<sequenceFlow id="f{i}" sourceRef="t{i}" targetRef="t{i + 1}"/>'
        for i in range(len(tasks) - 1)
    )

    return (
        '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">'
        f'<process id="p">{nodes}{flows}</process></definitions>'
    )


def kinds(measure: Any) -> dict[str, tuple[str, float]]:
    assert isinstance(measure, cbrkit.sim.AttributeValue)
    aggregator = measure.aggregator
    assert isinstance(aggregator, cbrkit.sim.Aggregator)
    by_class = {cls: kind for kind, cls in cbrkit.sim.MEASURES.by_kind.items()}

    return {
        name: (by_class[type(value)], aggregator.weights[name])
        for name, value in measure.attributes.items()
    }


def test_records_round_trip(tmp_path):
    path = tmp_path / "cars.jsonl"
    rows = [
        {
            "id": f"c{i}",
            "price": 100 * i,
            "fuel": fuel,
            "built": f"{2019 + i}-01-01",
            "tags": tags,
            "url": f"https://x/{i}",
        }
        for i, fuel, tags in [
            (1, "petrol", ["a", "b"]),
            (2, "diesel", ["b"]),
            (3, "petrol", []),
            (4, "diesel", None),
        ]
    ]
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows))
    source = cbrkit.io.detect(path).model_copy(update={"key_column": "id"})
    casebase = cbrkit.io.load(path, source)
    config = spec.infer(casebase, source)

    assert kinds(config.measure) == {
        "id": ("equality", 0.0),
        "price": ("linear", 1.0),
        "fuel": ("equality", 1.0),
        "built": ("date_linear", 1.0),
        "tags": ("jaccard", 1.0),
        "url": ("equality", 0.0),
    }

    sidecar = cbrkit.io.companion(path, "sidecar")
    spec.write(sidecar, config)

    assert spec.read(sidecar) == config
    assert spec.dumps(spec.read(sidecar)) == sidecar.read_text()

    query = {"price": 110, "fuel": "Petrol", "built": date(2020, 2, 1)}
    result = cbrkit.cycle.run_query(casebase, query, *config.steps())
    best = result.ranking[0]
    explanation = cbrkit.sim.explain(
        casebase[best], query, result.similarities[best], config.measure
    )

    assert best == "c1"
    assert sum(
        item.contribution or 0.0 for item in explanation.attributes.values()
    ) == pytest.approx(explanation.similarity)


def test_embedders_are_bound_by_name():
    async def embed(texts: Sequence[str]) -> list[NumpyArray]:
        return [np.array([float(len(text)), 1.0]) for text in texts]

    texts = {i: {"text": " ".join(["word"] * (10 + i))} for i in range(3)}
    config = spec.infer(texts, cbrkit.io.Source(format="json"), embedder="default")
    loaded = spec.Config.model_validate_json(spec.dumps(config))
    bound = spec.bind(loaded, {"default": embed}).measure

    assert isinstance(loaded.measure, AttributeValue)
    assert isinstance(bound, AttributeValue)
    assert isinstance(loaded.measure.attributes["text"], Embedding)
    embedding = bound.attributes["text"]
    assert isinstance(embedding, Embedding)
    assert loaded.measure.attributes["text"].embedder.func is None
    assert embedding.embedder.func is embed
    assert float(bound([(texts[0], texts[1])])[0]) < 1.0


def test_merge_keeps_decisions_and_marks_absent():
    source = cbrkit.io.Source(format="json")
    old = spec.infer({1: {"a": 1, "b": "x", "c": {"d": 1}}}, source)
    user = cbrkit.sim.numbers.Threshold(value=5)
    measure = old.measure
    assert isinstance(measure, AttributeValue)
    old = old.model_copy(
        update={
            "measure": dataclasses.replace(
                measure, attributes={**measure.attributes, "a": user}
            ),
            "provenance": old.provenance
            | {"/attributes/a": spec.Provenance(source="user", reason="tolerance")},
        }
    )
    cases = {1: {"a": "text", "c": {"d": 1, "e": True}, "f": 2.0}}
    merged = spec.infer(cases, source, old)
    assert isinstance(merged.measure, AttributeValue)
    nested = merged.measure.attributes["c"]
    assert isinstance(nested, AttributeValue)

    assert merged.measure.attributes["a"] == user
    assert merged.provenance["/attributes/a"].source == "user"
    assert merged.provenance["/attributes/b"].absent
    assert list(nested.attributes) == ["d", "e"]
    assert list(merged.measure.attributes) == ["a", "b", "c", "f"]
    assert spec.infer(cases, source, merged) == merged


def test_bpmn_graphs(tmp_path):
    pytest.importorskip("xsdata")
    (tmp_path / "a.bpmn").write_text(
        bpmn(
            [("startEvent", "Start"), ("userTask", "Check order"), ("endEvent", "End")]
        )
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.bpmn").write_text(
        bpmn(
            [
                ("startEvent", "Start"),
                ("serviceTask", "Ship goods"),
                ("endEvent", "End"),
            ]
        )
    )
    source = cbrkit.io.detect(tmp_path)
    casebase = cbrkit.io.load(tmp_path, source)
    config = spec.infer(casebase, source)
    graph = config.measure
    assert isinstance(graph, cbrkit.sim.graphs.LAP)
    table = graph.node_measure
    assert isinstance(table, cbrkit.sim.AttributeTable)
    node = kinds(table.fallback)

    assert list(casebase) == ["a.bpmn", "b/c.bpmn"]
    assert isinstance(graph.node_matcher, cbrkit.sim.graphs.AnyElement)
    assert node["type"][0] == "taxonomy" and node["process"][1] == 0

    query = cbrkit.model.graph.from_bpmn(
        cbrkit.model.graph.parse_bpmn(
            bpmn([("startEvent", "Start"), ("task", "check order")])
        )
    )
    result = cbrkit.cycle.run_query(casebase, query, *config.steps())
    explanation = cbrkit.sim.explain(
        casebase["a.bpmn"], query, result.similarities["a.bpmn"], config.measure
    )

    assert result.ranking[0] == "a.bpmn"
    assert explanation.graph is not None and explanation.graph.node_coverage == 1.0


def test_folders_skip_sidecars_and_hidden_files(tmp_path):
    (tmp_path / "a.json").write_text('{"price": 1}')
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "c.json").write_text('{"price": 2}')
    (tmp_path / "b" / "c.json.cbrkit.json").write_text("{}")
    (tmp_path / ".d.json").write_text("[]")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "e.bpmn").write_text("")
    source = cbrkit.io.detect(tmp_path)

    assert source == cbrkit.io.Source(format="json", folder=True)
    assert list(cbrkit.io.load(tmp_path, source)) == ["a.json", "b/c.json"]


def test_schema_reference_round_trips():
    config = spec.infer({1: {"x": 1}, 2: {"x": 2}}, cbrkit.io.Source(format="json"))

    assert "$schema" not in spec.dumps(config)

    referenced = config.model_copy(update={"schema_ref": "https://example.org/v2.json"})

    assert spec.Config.model_validate_json(spec.dumps(referenced)) == referenced


def test_edited_decisions_are_marked_as_user_decisions():
    previous = spec.infer(
        {
            1: {"x/y": 1.0, "o": {"p": "a", "q": 1.0}},
            2: {"x/y": 3.0, "o": {"p": "b", "q": 4.0}},
        },
        cbrkit.io.Source(format="json"),
    )
    data = previous.model_dump(mode="json")
    data["measure"]["aggregator"]["weights"]["x/y"] = 2.0
    data["measure"]["attributes"]["o"]["attributes"]["q"]["max_distance"] = 9.0
    edited = spec.mark_user(previous, spec.Config.model_validate(data))
    sources = {pointer: d.source for pointer, d in edited.provenance.items()}

    assert sources == {
        "": "heuristic",
        "/attributes/x~1y": "user",
        "/attributes/o": "heuristic",
        "/attributes/o/attributes/p": "heuristic",
        "/attributes/o/attributes/q": "user",
    }
    assert spec.mark_user(edited, edited) == edited


def test_refinement_is_validated():
    casebase = {
        i: {"size": size, "color": color}
        for i, (size, color) in enumerate([(1, "red"), (5, "blue"), (9, "red")])
    }
    config = spec.infer(casebase, cbrkit.io.Source(format="json"))
    patches = [
        {
            "path": "/attributes/size",
            "measure": {"kind": "exponential", "alpha": 0.5},
            "weight": 2,
            "reason": "small differences matter",
        },
        {
            "path": "/attributes/color",
            "measure": {
                "kind": "taxonomy",
                "taxonomy": {"name": "color", "children": ["red"]},
            },
            "reason": "incomplete",
        },
        {"path": "/attributes/weight", "weight": 1, "reason": "invented"},
        {"path": "", "measure": {"kind": "equality"}, "reason": "composite"},
    ]
    requests: list[tuple[str, dict[str, Any]]] = []

    async def model[T](prompts: Sequence[str], response_type: type[T], /) -> list[T]:
        adapter = TypeAdapter(response_type)
        requests.extend((prompt, adapter.json_schema()) for prompt in prompts)

        return [adapter.validate_python({"patches": patches}) for _ in prompts]

    refined, rejections = asyncio.run(
        spec.refine(
            config,
            casebase,
            cbrkit.sim.llm.LanguageModel("test-model", model),
            domain="paint shop",
        )
    )
    size = refined.provenance["/attributes/size"]
    prompt, schema = requests[0]

    assert schema["$defs"]["ConfigPatch"]["properties"]["path"]["enum"] == [
        "",
        "/attributes/size",
        "/attributes/color",
    ]
    assert "Critique" not in schema["$defs"]
    assert "paint shop" in prompt and "/attributes/color" in prompt
    assert kinds(refined.measure)["size"] == ("exponential", 2)
    assert (size.source, size.model) == ("llm", "test-model")
    assert isinstance(refined.measure, AttributeValue)
    assert isinstance(config.measure, AttributeValue)
    assert refined.measure.attributes["color"] == config.measure.attributes["color"]
    assert [rejection.path for rejection in rejections] == [
        "/attributes/color",
        "/attributes/weight",
        "",
    ]


def test_schema_is_derived_from_the_measures():
    schema = spec.Config.model_json_schema()
    linear = schema["$defs"]["Linear"]

    assert {
        "Measure",
        "Adaptation",
        "AttributeValue",
        "AttributeValueAdaptation",
        "LAP",
        "Taxonomy",
        "Embedding",
    } <= set(schema["$defs"])
    assert all(name.isalnum() for name in schema["$defs"])
    assert linear["properties"]["max_distance"]["description"].startswith(
        "Distance at which"
    )
    assert linear["properties"]["kind"]["const"] == "linear"


def test_custom_measures_need_no_second_definition():
    @serializable
    @dataclass(slots=True, frozen=True, kw_only=True)
    class Prefix:
        """Whether two strings share a prefix.

        Args:
            length: Length of the prefix.
        """

        length: int = 1
        kind: Literal["test_prefix"] = field(default="test_prefix", repr=False)

        def __call__(self, case: str, query: str) -> float:
            return float(case[: self.length] == query[: self.length])

    model = extend(spec.Config, Family("measure", (Prefix,)))
    config = spec.infer({1: "abc"}, cbrkit.io.Source(format="json"))
    measure = AttributeValue(attributes={"name": Prefix(length=2)})
    custom = model.model_validate(config.model_dump() | {"measure": measure})
    loaded = model.model_validate_json(spec.dumps(custom))

    assert loaded == custom and type(loaded) is model
    assert "Prefix" in model.model_json_schema()["$defs"]
    assert "Prefix" not in spec.Config.model_json_schema()["$defs"]

    with pytest.raises(ValidationError):
        spec.Config.model_validate_json(spec.dumps(custom))


def test_custom_functions_are_never_dropped_silently():
    config = spec.infer(
        {0: {"price": 1}, 1: {"price": 2}}, cbrkit.io.Source(format="json")
    )
    measure = config.measure
    assert isinstance(measure, cbrkit.sim.AttributeValue)
    custom = config.model_copy(
        update={"measure": dataclasses.replace(measure, value_getter=getattr)}
    )

    with pytest.raises(PydanticSerializationError, match="value_getter"):
        spec.dumps(custom)


def test_sequence_and_graph_measures_round_trip():
    from cbrkit.sim.collections import TWED, SequenceMapping, SmithWaterman
    from cbrkit.sim.generic import Equality
    from cbrkit.sim.graphs import DFS, VF2Networkx
    from cbrkit.sim.wrappers import Combine

    for measure in [
        TWED(stiffness=0.5),
        SmithWaterman(element_measure=Equality()),
        SequenceMapping(element_measure=Equality(), exact=True),
        DFS(node_measure=Equality(), timeout=1.0),
        VF2Networkx(node_measure=Equality()),
        Combine(measures={"a": Equality()}),
    ]:
        dumped = MEASURE.dump_python(measure, mode="json")

        assert MEASURE.validate_python(dumped) == measure


def test_cycle_sections_round_trip_and_validate():
    config = spec.infer({0: {"price": 1.0}}, cbrkit.io.Source(format="csv"))
    config = config.model_copy(
        update={
            "reuse": cbrkit.reuse.Reuser(
                adaptation=cbrkit.adapt.AttributeValueAdaptation(
                    attributes={"price": cbrkit.adapt.numbers.Aggregate()}
                )
            ),
            "retain": cbrkit.retain.Retainer(),
            "maintain": cbrkit.maintain.Maintainer(
                competence=0.9, remove=frozenset({"auxiliary"})
            ),
            "limits": spec.Limits(dry_run=True),
            "providers": spec.Providers(embedders={"text": "openai:small"}),
        }
    )

    assert spec.Config.model_validate_json(spec.dumps(config)) == config

    with pytest.raises(ValueError, match="written"):
        spec.Config.model_validate(config.model_dump() | {"source": {"format": "xlsx"}})

    with pytest.raises(ValueError, match="consistency"):
        cbrkit.maintain.Maintainer(competence=0.9, consistency=0.5)

    with pytest.raises(ValueError, match="extra"):
        spec.case_model(config).model_validate({"price": 1.0, "color": "red"})


def test_file_storage_keeps_concurrent_writes(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("id,price\na,1\n")
    source = cbrkit.io.Source(format="csv", key_column="id")
    writers = [cbrkit.io.FileStorage(path=path, source=source) for _ in range(4)]

    async def add() -> None:
        await asyncio.gather(
            *(
                writers[i % 4].patch_index(
                    upsert={f"k{i}": {"id": f"k{i}", "price": i}}
                )
                for i in range(20)
            )
        )

    asyncio.run(add())
    cases = asyncio.run(cbrkit.io.FileStorage(path=path, source=source).get_index())

    assert len(cases) == 21
    assert cases["k7"]["price"] == 7


def test_file_storage_patches_keep_the_keys(tmp_path):
    keyed = cbrkit.io.Source(format="csv", key_column="id")
    path = tmp_path / "cases.csv"
    path.write_bytes(b"\xef\xbb\xbfid,price\na,1\nb,2\n")
    asyncio.run(
        cbrkit.io.FileStorage(path=path, source=keyed).patch_index(
            upsert={"b": {"id": "b", "price": 3}}
        )
    )

    assert path.read_bytes() == b"\xef\xbb\xbfid,price\na,1\nb,3\n"

    positional = cbrkit.io.Source(format="jsonl")
    path = tmp_path / "cases.jsonl"
    path.write_text('{"a": 1}\n')

    with pytest.raises(cbrkit.errors.Refused):
        asyncio.run(
            cbrkit.io.FileStorage(path=path, source=positional).patch_index(
                insert={5: {"a": 2}}
            )
        )


def test_stored_cases_allow_unmeasured_nested_fields():
    config = spec.infer({0: {"car": {"price": 1.0}}}, cbrkit.io.Source(format="jsonl"))
    case = {"car": {"price": 1.0, "color": "red"}}
    spec.case_model(config, stored=True).model_validate(case)

    with pytest.raises(ValidationError):
        spec.case_model(config).model_validate(case)


def test_measures_without_attributes_score_every_pair():
    results = AttributeValue(attributes={})([({}, {}), ({}, {})])

    assert [result.similarities for result in results] == [{}, {}]


@pytest.mark.parametrize(
    "source",
    [
        cbrkit.io.Source(format="csv"),
        cbrkit.io.Source(format="csv", key_column="id"),
        cbrkit.io.Source(format="parquet", key_column="id"),
        cbrkit.io.Source(format="json"),
        cbrkit.io.Source(format="jsonl"),
        cbrkit.io.Source(format="yaml", key_column="id"),
        cbrkit.io.Source(format="json", folder=True),
    ],
    ids=lambda source: f"{source.format}-{source.key_column}-{source.folder}",
)
def test_writable_formats_round_trip(tmp_path, source):
    path = tmp_path / ("cases" if source.folder else f"cases.{source.format}")
    keys = (
        ["a.json", "b.json", "c.json"]
        if source.folder
        else ["a", "b", "c"]
        if source.key_column
        else [0, 1, 2]
    )
    cases = {
        key: {"price": 10 * i, "fuel": f"f{i}"}
        | ({source.key_column: key} if source.key_column else {})
        for i, key in enumerate(keys)
    }
    cbrkit.io.save(path, source, cases)

    assert dict(cbrkit.io.load(path, source)) == cases

    # cases keyed by position can only be added, deleting one would shift the others
    keyed = source.key_column is not None or source.folder
    storage = cbrkit.io.FileStorage(path=path, source=source)
    asyncio.run(storage.put_index({k: cases[k] for k in keys[:2]}))
    added = asyncio.run(
        storage.patch_index(
            delete=keys[:1] if keyed else None, add=[{"price": 99, "fuel": "new"}]
        )
    )
    reloaded = dict(
        asyncio.run(cbrkit.io.FileStorage(path=path, source=source).get_index())
    )

    assert len(added) == 1
    assert reloaded[added[0]]["price"] == 99
    assert (keys[0] in reloaded) is not keyed


def test_render_matches_the_file_storage(tmp_path):
    source = cbrkit.io.Source(format="json", folder=True)
    cbrkit.io.save(tmp_path, source, {"a.json": {"x": 1}, "b.json": {"x": 2}})
    stored = cbrkit.io.load(tmp_path, source)
    changes = cbrkit.model.Changes[str, dict[str, int]](
        items=[
            cbrkit.model.Update(key="a.json", value={"x": 3}, reason=""),
            cbrkit.model.Remove(key="b.json", reason=""),
        ],
        expected=stored,
    )

    assert cbrkit.io.render(tmp_path, source, stored, changes) == {
        tmp_path / "a.json": b'{\n  "x": 3\n}',
        tmp_path / "b.json": None,
    }


def test_folder_storage_reloads_changed_members(tmp_path):
    source = cbrkit.io.Source(format="json", folder=True)
    cbrkit.io.save(tmp_path, source, {"a.json": {"x": 1}, "sub/b.json": {"x": 2}})
    storage = cbrkit.io.FileStorage(path=tmp_path, source=source)
    asyncio.run(storage.get_index())
    (tmp_path / "sub" / "b.json").write_text('{"x": 30}')

    assert asyncio.run(storage.get_index())["sub/b.json"] == {"x": 30}
