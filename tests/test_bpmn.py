import pickle
from dataclasses import replace

import pydantic_core
import pytest

import cbrkit
from cbrkit.helpers import frozendict

pytest.importorskip("xsdata")

from cbrkit.model.graph import (
    bpmn_document,
    from_bpmn,
    from_json,
    parse_bpmn,
    render_bpmn,
)

BPMN = """<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL"
    xmlns:camunda="http://camunda.org/schema/1.0/bpmn" targetNamespace="urn:test">
  <collaboration id="c">
    <participant id="shop" name="Shop" processRef="order" />
    <participant id="customer" name="Customer" />
    <messageFlow id="m" sourceRef="customer" targetRef="paid" />
  </collaboration>
  <process id="order">
    <laneSet id="ls">
      <lane id="l" name="Sales">
        <flowNodeRef>start</flowNodeRef>
        <flowNodeRef>check</flowNodeRef>
      </lane>
    </laneSet>
    <startEvent id="start" name="Order received" />
    <dataObjectReference id="data" name="Order" />
    <userTask id="check" name="Check order" camunda:assignee="ann">
      <property id="input" />
      <dataInputAssociation id="in">
        <sourceRef>data</sourceRef>
        <targetRef>input</targetRef>
      </dataInputAssociation>
      <dataOutputAssociation id="out">
        <targetRef>data</targetRef>
      </dataOutputAssociation>
    </userTask>
    <boundaryEvent id="late" attachedToRef="check">
      <timerEventDefinition id="td"><timeDuration>PT1H</timeDuration></timerEventDefinition>
    </boundaryEvent>
    <subProcess id="ship" name="Ship">
      <intermediateCatchEvent id="paid">
        <messageEventDefinition messageRef="payment" />
      </intermediateCatchEvent>
    </subProcess>
    <sequenceFlow id="f1" sourceRef="start" targetRef="check" />
    <sequenceFlow id="f2" sourceRef="check" targetRef="ship">
      <conditionExpression language="python">paid</conditionExpression>
    </sequenceFlow>
  </process>
</definitions>
"""


def test_bpmn_details() -> None:
    g = from_bpmn(parse_bpmn(BPMN))

    assert g.nodes["late"].value["event"] == "timer"
    assert g.nodes["late"].value["timeDuration"] == "PT1H"
    assert g.nodes["paid"].value["event"] == "message"
    assert g.nodes["paid"].value["messageRef"] == "payment"
    assert g.nodes["paid"].value["parent"] == "ship"
    assert "parent" not in g.nodes["check"].value
    assert (
        g.nodes["check"].value["{http://camunda.org/schema/1.0/bpmn}assignee"] == "ann"
    )
    assert g.edges["f2"].value["language"] == "python"
    assert (g.edges["in"].source.key, g.edges["in"].target.key) == ("data", "check")
    assert (g.edges["out"].source.key, g.edges["out"].target.key) == ("check", "data")


def test_bpmn_round_trip() -> None:
    g = from_bpmn(parse_bpmn(BPMN))
    written = from_bpmn(parse_bpmn(render_bpmn(bpmn_document(g))))

    assert written == g
    assert bpmn_document(written) == bpmn_document(g)
    assert bpmn_document(pickle.loads(pickle.dumps(g))) == bpmn_document(g)
    assert cbrkit.model.graph.to_dict(written) == cbrkit.model.graph.to_dict(g)


def test_bpmn_write_needs_document() -> None:
    g = from_bpmn(parse_bpmn(BPMN))
    check = replace(
        g.nodes["check"], value=frozendict(g.nodes["check"].value, name="Ship")
    )
    changed = replace(g, nodes=frozendict(g.nodes, check=check))
    loaded = from_json(pydantic_core.to_json(g))

    assert loaded.value == {
        "type": "definitions",
        "name": None,
        "targetNamespace": "urn:test",
    }

    for graph in (loaded, changed):
        with pytest.raises(ValueError, match="Only graphs read from BPMN"):
            bpmn_document(graph)
