from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

import cbrkit

JSON = cbrkit.io.Source(format="json")

type Graph[N] = cbrkit.model.graph.Graph[str, N, None, None]

ALGORITHMS: dict[
    str,
    Callable[
        ..., cbrkit.typing.AnySimFunc[Graph[str | int], cbrkit.model.GraphSim[str]]
    ],
] = {
    "astar1": partial(
        cbrkit.sim.graphs.Astar,
        heuristic=cbrkit.sim.graphs.astar.Unmapped(),
        selection=cbrkit.sim.graphs.astar.InOrder(),
    ),
    "astar2": partial(
        cbrkit.sim.graphs.Astar,
        heuristic=cbrkit.sim.graphs.astar.BestPair(),
        selection=cbrkit.sim.graphs.astar.EdgesFirst(),
    ),
    "astar3": partial(
        cbrkit.sim.graphs.Astar,
        heuristic=cbrkit.sim.graphs.astar.BestOpenPair(),
        selection=cbrkit.sim.graphs.astar.MostConstrained(),
    ),
    "astar4": partial(
        cbrkit.sim.graphs.Astar,
        heuristic=cbrkit.sim.graphs.astar.OpenAssignment(),
        selection=cbrkit.sim.graphs.astar.AssignmentCost(),
    ),
    "brute_force": cbrkit.sim.graphs.BruteForce,
    "dfs": cbrkit.sim.graphs.DFS,
    "greedy": cbrkit.sim.graphs.Greedy,
    "lap": partial(cbrkit.sim.graphs.LAP, edge_handling="ignore"),
    "lap_greedy": partial(cbrkit.sim.graphs.LAP, edge_handling="greedy"),
    "lap_optimal": partial(cbrkit.sim.graphs.LAP, edge_handling="optimal"),
    "qap": cbrkit.sim.graphs.QAP,
    "vf2_networkx": cbrkit.sim.graphs.VF2Networkx,
    "vf2_rustworkx": cbrkit.sim.graphs.VF2Rustworkx,
}


@dataclass
class Setup[N]:
    casebase: cbrkit.typing.Casebase[str, Graph[N]]
    query: Graph[N]
    node_measure: cbrkit.typing.AnySimFunc[N, cbrkit.typing.Float]
    node_matcher: cbrkit.sim.graphs.ElementMatcher[N]
    baseline: cbrkit.typing.Casebase[str, Any]


@pytest.fixture(scope="session")
def setup_v1() -> Setup[str | int]:
    return Setup(
        casebase={
            key: cbrkit.model.graph.from_dict(value)
            for key, value in cbrkit.io.load("./data/graphs-v1.json", JSON).items()
        },
        query=cbrkit.model.graph.from_dict(
            {
                "nodes": {
                    "node1": "A string value",
                    "node2": 42,
                },
                "edges": {
                    "edge1": {"source": "node1", "target": "node2", "value": None}
                },
                "value": None,
            }
        ),
        node_measure=cbrkit.sim.AttributeTable(
            entries={
                "str": cbrkit.sim.generic.Equality(),
                "int": cbrkit.sim.numbers.LinearInterval(lower=0, upper=200),
            },
        ),
        node_matcher=lambda x, y: type(x) is type(y),
        baseline={},
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS.keys())
def test_v1(setup_v1, algorithm) -> None:
    func = ALGORITHMS[algorithm]
    retriever = cbrkit.retrieve.Retriever(
        measure=func(
            node_measure=setup_v1.node_measure,
            node_matcher=setup_v1.node_matcher,
        )
    )
    result = cbrkit.cycle.run_query(
        setup_v1.casebase,
        setup_v1.query,
        retriever,
    )

    assert result.similarities["first"].value == pytest.approx(1.0)
    assert result.similarities["second"].value < 1.0


@pytest.fixture(scope="session")
def setup_v2() -> Setup[dict[str, str]]:
    return Setup(
        casebase={
            key: cbrkit.model.graph.from_dict(value)
            for key, value in cbrkit.io.load("./data/graphs-v2.json", JSON).items()
        },
        query=cbrkit.model.graph.from_dict(
            {
                "nodes": {
                    "q1": {"type": "Person", "text": "Alice"},
                    "q2": {"type": "Company", "text": "Acme Corp"},
                },
                "edges": {"qe1": {"source": "q1", "target": "q2", "value": None}},
                "value": None,
            }
        ),
        node_measure=cbrkit.sim.AttributeValue(
            attributes={
                "text": lambda x, y: 1.0 if x == y else 0.2,
            },
            aggregator=cbrkit.sim.Aggregator(),
        ),
        node_matcher=lambda y, x: y["type"] == x["type"],
        baseline=cbrkit.io.load("./data/graphs-v2-baseline.json", JSON),
    )


@pytest.mark.parametrize("algorithm", ALGORITHMS.keys())
def test_v2(setup_v2, algorithm):
    func = ALGORITHMS[algorithm]
    retriever = cbrkit.retrieve.Retriever(
        measure=func(
            node_measure=setup_v2.node_measure,
            node_matcher=setup_v2.node_matcher,
        )
    )
    result = cbrkit.cycle.run_query(
        setup_v2.casebase,
        setup_v2.query,
        retriever,
    )

    for key, sim in result.similarities.items():
        baseline = setup_v2.baseline[key]

        if algorithm.startswith("vf2") and not baseline["edge_mapping"]:
            # If there is no edge mapping in the baseline, the algorithm does not find any mapping
            continue

        # assert sim.value == pytest.approx(baseline["value"]), key
        assert sim.mapping == baseline["node_mapping"], key
        assert sim.edge_mapping == baseline["edge_mapping"], key
        assert {
            node: float(value) for node, value in sim.similarities.items()
        } == pytest.approx(baseline["node_similarities"]), key
        assert all(
            isinstance(value, cbrkit.model.AttributeValueSim)
            for value in sim.similarities.values()
        ), key
        assert sim.edge_similarities == pytest.approx(baseline["edge_similarities"]), (
            key
        )


BPMN = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" id="d">
  <collaboration id="c">
    <participant id="shop" name="Shop" processRef="order" />
    <participant id="customer" name="Customer" />
    <messageFlow id="m" sourceRef="customer" targetRef="start" />
  </collaboration>
  <process id="order">
    <laneSet id="ls">
      <lane id="l" name="Sales">
        <flowNodeRef>start</flowNodeRef>
        <flowNodeRef>gateway</flowNodeRef>
      </lane>
    </laneSet>
    <startEvent id="start" name="Order received" />
    <exclusiveGateway id="gateway" />
    <subProcess id="ship" name="Ship">
      <task id="pack" name="Pack" />
    </subProcess>
    <sequenceFlow id="f1" sourceRef="start" targetRef="gateway" />
    <sequenceFlow id="f2" sourceRef="gateway" targetRef="ship">
      <conditionExpression>paid</conditionExpression>
    </sequenceFlow>
  </process>
</definitions>
"""


def test_bpmn_loader() -> None:
    pytest.importorskip("xsdata")
    g = cbrkit.model.graph.from_bpmn(cbrkit.model.graph.parse_bpmn(BPMN))

    assert set(g.nodes) == {"start", "gateway", "ship", "pack", "customer"}
    assert g.nodes["start"].value == {
        "type": "startEvent",
        "name": "Order received",
        "process": "order",
        "pool": "Shop",
        "lane": "Sales",
    }
    assert g.nodes["customer"].value["type"] == "participant"
    assert g.edges["f2"].value["condition"] == "paid"
    assert g.edges["m"].value["type"] == "messageFlow"


def test_bpmn_loader_rejects_entities() -> None:
    pytest.importorskip("xsdata")
    document = (
        '<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        '<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">&x;</definitions>'
    )

    with pytest.raises(ValueError, match="Document type"):
        cbrkit.model.graph.parse_bpmn(document)
