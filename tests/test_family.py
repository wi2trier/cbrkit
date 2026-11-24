"""Every built-in kind round-trips through a sidecar file read in a fresh process."""

import dataclasses
import json
import subprocess
import sys
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

import cbrkit
from cbrkit import spec
from cbrkit.adapt import ADAPTATIONS
from cbrkit.schema import Family, class_docs
from cbrkit.sim import MEASURES

EQUALITY = {"kind": "equality"}
MEASURE_EXAMPLES = {
    "astar": {"node_measure": EQUALITY},
    "attribute_table": {"attribute": "type", "entries": {}},
    "attribute_value": {
        "attributes": {"price": {"kind": "linear", "max_distance": 10}}
    },
    "brute_force": {"node_measure": EQUALITY},
    "combine": {"measures": {"equal": EQUALITY}},
    "date_linear": {"max_distance": "P1D"},
    "dfs": {"node_measure": EQUALITY},
    "dtw": {},
    "embedding": {
        "embedder": "default",
        "metric": {"kind": "cosine"},
    },
    "equality": {},
    "exponential": {},
    "glob": {},
    "graph_dtw": {"node_measure": EQUALITY},
    "graph_smith_waterman": {"node_measure": EQUALITY},
    "graph_twed": {"node_measure": EQUALITY},
    "greedy": {"node_measure": EQUALITY},
    "indel": {},
    "isolated_mapping": {"element_measure": EQUALITY},
    "jaccard": {},
    "jaro": {},
    "jaro_winkler": {},
    "lap": {"node_measure": EQUALITY},
    "levenshtein": {},
    "linear": {"max_distance": 10},
    "linear_interval": {"lower": 0, "upper": 10},
    "ngram": {"n": 2},
    "optimal_mapping": {"element_measure": EQUALITY},
    "qap": {"node_measure": EQUALITY},
    "regex": {},
    "sequence_correctness": {},
    "sequence_mapping": {"element_measure": EQUALITY},
    "sigmoid": {},
    "smith_waterman": {},
    "static": {"value": 0.5},
    "table": {"entries": []},
    "taxonomy": {"taxonomy": {"name": "root"}},
    "threshold": {"value": 1},
    "twed": {},
    "type_equality": {},
    "vf2_networkx": {"node_measure": EQUALITY},
    "vf2_rustworkx": {"node_measure": EQUALITY},
    "word_coverage": {},
    "critique": {"model": "judge", "instructions": "Judge the solution."},
}
ADAPTATION_EXAMPLES = {
    "aggregate": {},
    "attribute_value": {"attributes": {"price": {"kind": "aggregate"}}},
    "null": {},
    "pipe": {"adaptations": [{"kind": "null"}]},
    "regex": {"case_pattern": "a", "query_pattern": "b", "replacement": "c"},
    "rewrite": {
        "model": "writer",
        "instructions": "Adapt the case.",
        "attributes": ["price"],
    },
    "union": {},
}
READ = """
import sys
from cbrkit import spec
for path in sys.argv[1:]:
    print(spec.dumps(spec.read(path)), end="\\0")
"""


def sidecar(measure: dict[str, Any], **sections: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": {"format": "json"},
        "fingerprint": "",
        "measure": measure,
        **sections,
    }


def test_every_builtin_kind_round_trips_in_a_fresh_process(tmp_path):
    assert set(MEASURE_EXAMPLES) == set(MEASURES.by_kind)
    assert set(ADAPTATION_EXAMPLES) == set(ADAPTATIONS.by_kind)

    configs = [
        sidecar({"kind": kind, **fields}) for kind, fields in MEASURE_EXAMPLES.items()
    ] + [
        sidecar(EQUALITY, reuse={"adaptation": {"kind": kind, **fields}})
        for kind, fields in ADAPTATION_EXAMPLES.items()
    ]
    paths = []

    for index, config in enumerate(configs):
        path = tmp_path / f"{index}.cbrkit.json"
        path.write_text(json.dumps(config))
        paths.append(str(path))

    output = subprocess.run(
        [sys.executable, "-c", READ, *paths], capture_output=True, text=True, check=True
    ).stdout
    dumped = output.split("\0")[:-1]

    assert dumped == [spec.dumps(spec.read(path)) for path in paths]
    assert [json.loads(text)["measure"]["kind"] for text in dumped] == [
        config["measure"]["kind"] for config in configs
    ]


def test_every_kind_documents_its_fields():
    modules = [
        cbrkit.sim,
        cbrkit.adapt,
        cbrkit.sim.pooling,
        cbrkit.sim.taxonomy,
        cbrkit.sim.graphs,
        cbrkit.sim.graphs.astar,
        cbrkit.sim.embed.metrics,
    ]
    families = {
        value.name: value
        for module in modules
        for value in vars(module).values()
        if isinstance(value, Family)
    }.values()
    # classes that serialize outside a family
    others = [
        cbrkit.sim.Aggregator,
        cbrkit.model.AttributeValueSim,
        cbrkit.model.ChangeSim,
        cbrkit.model.CritiqueSim,
        cbrkit.model.GraphSim,
        cbrkit.model.SequenceSim,
        cbrkit.retrieve.Retriever,
        cbrkit.reuse.Reuser,
        cbrkit.revise.Reviser,
        cbrkit.retain.Retainer,
        cbrkit.maintain.Maintainer,
        cbrkit.eval.common.Metric,
    ]
    undocumented = [
        f"{cls.__name__}.{f.name}"
        for cls in [*(c for family in families for c in family.members), *others]
        for f in dataclasses.fields(cls)
        if f.init and f.name != "kind" and f.name not in class_docs(cls)[1]
    ]

    assert len(families) == 10
    assert undocumented == []


def test_named_family_needs_its_members() -> None:
    with pytest.raises(TypeError, match="No measure members"):

        class Holder(BaseModel):
            measure: Annotated[Any, Family("measure")]
