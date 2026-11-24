from collections.abc import Iterable, Iterator, Mapping
from typing import Any
from xml.etree import ElementTree

from pydantic_core import SchemaSerializer, core_schema
from xsdata.formats.converter import converter
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.parsers.config import ParserConfig
from xsdata.formats.dataclass.parsers.handlers import XmlEventHandler
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig
from xsdata.models.enums import Namespace
from xsdata.utils.namespaces import local_name

from ...helpers import frozendict
from ._bpmn_schema import model as bpmn
from ._bpmn_types import BPMN_NODES, bpmn_class
from ._graph import Edge, Graph, Node

__all__ = [
    "BPMN_NODES",
    "BpmnEdge",
    "BpmnGraph",
    "BpmnNode",
    "BpmnValue",
    "bpmn",
    "bpmn_class",
    "bpmn_document",
    "bpmn_event_class",
    "bpmn_problems",
    "from_bpmn",
    "parse_bpmn",
    "render_bpmn",
]

type BpmnValue = frozendict[str, str | None]
type BpmnNode = Node[str, BpmnValue]
type BpmnEdge = Edge[str, BpmnValue, BpmnValue]
type BpmnGraph = Graph[str, BpmnValue, BpmnValue, bpmn.Definitions]

BPMN_NAMESPACE = "http://www.omg.org/spec/BPMN/20100524/MODEL"
_CONTEXT = XmlContext()


def _create[T](cls: type[T], params: dict[str, Any]) -> T:
    """Instantiates a class, with `None` for required values that are missing.

    BPMN documents in the wild often omit them, such as the target namespace.
    """
    missing = {
        var.name: None
        for var in _CONTEXT.build(cls).get_all_vars()
        if var.required and var.name not in params
    }

    return cls(**params, **missing)


# The native handler binds the element tree that rejected document type declarations.
_PARSER = XmlParser(
    config=ParserConfig(class_factory=_create),
    context=_CONTEXT,
    handler=XmlEventHandler,
)
_SERIALIZER = XmlSerializer(
    context=_CONTEXT,
    config=SerializerConfig(indent="  ", ignore_default_attributes=True),
)
_REFERENCES = frozenset({"id", "sourceRef", "targetRef"})
"""Attributes that the graph structure records instead of the values."""
_EVENT_DEFINITION = "EventDefinition"
"""The suffix of the elements of event definitions, which the `event` of a record leaves out."""

type _Element = bpmn.TBaseElement | bpmn.TDefinitions
type _Record = dict[str, str | None]


class _NoDoctypeBuilder(ElementTree.TreeBuilder):
    """Rejects document type declarations, which rules out entity expansion."""

    def doctype(self, name: str, pubid: str, system: str) -> None:
        raise ValueError("Document type declarations are not allowed")


def _type(element: _Element) -> str:
    """The local name of the element of a generated class, the `type` of its record."""
    return local_name(_CONTEXT.build(type(element)).qname)


def bpmn_event_class(value: Mapping[str, str | None]) -> type[bpmn.TBaseElement] | None:
    """The generated class of the event definition a node value records, None for a plain event."""
    event = value.get("event")

    return None if event is None else bpmn_class(event + _EVENT_DEFINITION)


def _text(expression: bpmn.TExpression) -> str:
    return "".join(item for item in expression.content if isinstance(item, str)).strip()


def _bpmn_record(element: _Element, **extra: str | None) -> _Record:
    """Collects the type, name and all attributes of an element that differ from their defaults."""
    attributes: _Record = {}

    for var in _CONTEXT.build(type(element)).get_attribute_vars():
        value = getattr(element, var.name)

        if var.is_attributes:
            attributes.update(value)
        elif (
            var.local_name not in _REFERENCES
            and value is not None
            and value != var.default
        ):
            attributes[var.local_name] = converter.serialize(value)

    return {"type": _type(element), "name": None, **attributes, **extra}


def _bpmn_event(element: bpmn.TFlowElement) -> _Record:
    """Collects the kind, attributes and expressions of the first event definition of an event."""
    if not isinstance(element, bpmn.TCatchEvent | bpmn.TThrowEvent):
        return {}

    for definition in element.event_definition:
        expressions: _Record = {}

        for var in _CONTEXT.build(type(definition)).get_element_vars():
            value = getattr(definition, var.name)

            if isinstance(value, bpmn.TExpression):
                choice = var.find_value_choice(value, True) if var.is_elements else var
                expressions[(choice or var).local_name] = _text(value)

        return {
            "event": _type(definition).removesuffix(_EVENT_DEFINITION),
            **{
                key: value
                for key, value in _bpmn_record(definition).items()
                if key not in ("type", "name")
            },
            **expressions,
        }

    return {}


def _bpmn_flow_edge(
    flow: bpmn.TSequenceFlow | bpmn.TMessageFlow, nodes: Mapping[str, BpmnNode]
) -> BpmnEdge:
    """Connects the nodes of a sequence or message flow, with its condition."""
    source, target = str(flow.source_ref), str(flow.target_ref)

    if source not in nodes or target not in nodes:
        raise ValueError(
            f"Flow {flow.id!r} connects unknown elements {source!r} and {target!r}"
        )

    condition = (
        flow.condition_expression if isinstance(flow, bpmn.TSequenceFlow) else None
    )
    extra: _Record = (
        {}
        if condition is None
        else {
            "condition": _text(condition),
            "language": condition.language
            if isinstance(condition, bpmn.TFormalExpression)
            else condition.other_attributes.get("language"),
        }
    )

    return Edge(
        value=frozendict(_bpmn_record(flow, **extra)),
        key=flow.id or "",
        source=nodes[source],
        target=nodes[target],
    )


def _bpmn_lanes(lane_sets: Iterable[bpmn.TLaneSet]) -> Iterator[tuple[str, str | None]]:
    """The flow nodes of lanes with the lane names, nested lanes after their parents."""
    for lane_set in lane_sets:
        for lane in lane_set.lane:
            for ref in lane.flow_node_ref:
                yield ref.strip(), lane.name

            if lane.child_lane_set is not None:
                yield from _bpmn_lanes([lane.child_lane_set])


def parse_bpmn(source: str | bytes) -> bpmn.Definitions:
    """Read a BPMN 2.0 XML document, bound to the classes generated from the OMG BPMN 2.0 schemas.

    Document type declarations are rejected, so no entities are ever expanded.

    Raises:
        ValueError: If the document has a document type declaration or does not fit the schemas.
        xml.etree.ElementTree.ParseError: If the document is no XML.
    """
    parser = ElementTree.XMLParser(target=_NoDoctypeBuilder())
    parser.feed(source)

    return _PARSER.parse(parser.close(), bpmn.Definitions)


def from_bpmn(document: bpmn.Definitions) -> BpmnGraph:
    """Construct a graph from a BPMN 2.0 document, such as one read by `parse_bpmn`.

    The flow nodes (events, activities, gateways and data references) of all
    processes become nodes, including those nested in subprocesses, and sequence
    and message flows become edges.
    Each node and edge value is a record of its BPMN `type`, its `name` and all other
    XML attributes that differ from their schema defaults, nodes also carry the
    `process`, `pool` and `lane` they belong to and sequence flows their `condition`
    and its `language`.
    Extension attributes keep their namespace in Clark notation, such as
    `{http://camunda.org/schema/1.0/bpmn}assignee`, unqualified ones their name.
    Nodes nested in a subprocess carry its id as `parent`, and events with an event
    definition carry its kind as `event` (such as `message`, `timer`, or `error`)
    together with its attributes (`messageRef`, `errorRef`, ...)
    and expressions (`timeDuration`, `timeDate`, `timeCycle`, `condition`).
    Data input and output associations between data references and activities
    become edges from the data to the activity and from the activity to the data.
    A pool without a process that is connected by a message flow becomes a node of
    type `participant`.
    Node and edge values are immutable, and the graph value is the document,
    which `bpmn_document` returns for writing as long as the graph is unchanged.
    Pydantic serializes the document as its record, so a graph loaded from JSON has none.

    Raises:
        ValueError: If a flow connects unknown elements.

    Examples:
        >>> g = from_bpmn(parse_bpmn('''
        ...     <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        ...       <process id="p" name="Order">
        ...         <startEvent id="s"/>
        ...         <userTask id="t" name="Check order"/>
        ...         <sequenceFlow id="f" sourceRef="s" targetRef="t"/>
        ...       </process>
        ...     </definitions>
        ... '''))
        >>> dict(g.nodes["t"].value)
        {'type': 'userTask', 'name': 'Check order', 'process': 'p', 'pool': None, 'lane': None}
        >>> g.edges["f"].source.key, g.edges["f"].value["type"]
        ('s', 'sequenceFlow')
    """
    processes = [item for item in document.choice if isinstance(item, bpmn.TProcess)]
    participants = [
        participant
        for item in document.choice
        if isinstance(item, bpmn.TCollaboration)
        for participant in item.participant
    ]
    pools: dict[str | None, str | None] = {
        str(participant.process_ref): participant.name
        for participant in participants
        if participant.process_ref is not None
    }
    lanes = dict(_bpmn_lanes(lane_set for p in processes for lane_set in p.lane_set))
    nodes: dict[str, BpmnNode] = {}
    flows: list[bpmn.TSequenceFlow | bpmn.TMessageFlow] = []
    associations: list[tuple[str, str | None, str, str, str]] = []

    def collect(
        container: bpmn.TProcess | bpmn.TSubProcess,
        process: str | None,
        parent: str | None,
    ) -> None:
        for element in container.flow_element:
            if isinstance(element, BPMN_NODES) and element.id is not None:
                key = element.id
                nested = {} if parent is None else {"parent": parent}
                nodes[key] = Node(
                    value=frozendict(
                        _bpmn_record(
                            element,
                            process=process,
                            pool=pools.get(process),
                            lane=lanes.get(key),
                            **nested,
                            **_bpmn_event(element),
                        )
                    ),
                    key=key,
                )

                if isinstance(element, bpmn.TActivity | bpmn.TThrowEvent):
                    associations.extend(
                        ("dataInputAssociation", association.id, key, ref.strip(), key)
                        for association in element.data_input_association
                        for ref in association.source_ref[:1]
                    )

                if isinstance(element, bpmn.TActivity | bpmn.TCatchEvent):
                    associations.extend(
                        (
                            "dataOutputAssociation",
                            association.id,
                            key,
                            key,
                            association.target_ref.strip(),
                        )
                        for association in element.data_output_association
                    )

            if isinstance(element, bpmn.TSequenceFlow):
                flows.append(element)

            if isinstance(element, bpmn.TSubProcess):
                collect(element, process, element.id)

    for process in processes:
        collect(process, process.id, None)

    for participant in participants:
        if participant.process_ref is None and participant.id is not None:
            nodes[participant.id] = Node(
                value=frozendict(_bpmn_record(participant)), key=participant.id
            )

    flows.extend(
        flow
        for item in document.choice
        if isinstance(item, bpmn.TCollaboration)
        for flow in item.message_flow
    )
    edges = [_bpmn_flow_edge(flow, nodes) for flow in flows]

    # Associations with properties or data inputs of an activity have no node to connect.
    for index, (kind, key, activity, start, end) in enumerate(associations):
        if start in nodes and end in nodes:
            edges.append(
                Edge(
                    value=frozendict(type=kind, name=None),
                    key=key or f"{activity}-association-{index}",
                    source=nodes[start],
                    target=nodes[end],
                )
            )

    return Graph.build(nodes.values(), edges, document)


# pydantic serializes the document as its record, also as a graph value in fields typed `Any`
setattr(  # noqa: B010
    bpmn.Definitions,
    "__pydantic_serializer__",
    SchemaSerializer(
        core_schema.is_instance_schema(
            bpmn.Definitions,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _bpmn_record, info_arg=False
            ),
        )
    ),
)


def bpmn_document(graph: BpmnGraph) -> bpmn.Definitions:
    """The BPMN 2.0 document a graph was constructed from by `from_bpmn`.

    The graph has to be the one of its document, which is checked by reading the document again.

    Raises:
        ValueError: If the graph was not read from BPMN, such as one built by hand
            or loaded from JSON, or changed after it was read.
    """
    document: object = graph.value

    if not isinstance(document, bpmn.Definitions) or from_bpmn(document) != graph:
        raise ValueError(
            "Only graphs read from BPMN by from_bpmn and unchanged since can be written as BPMN"
        )

    return document


def render_bpmn(document: bpmn.Definitions) -> str:
    """Write a BPMN 2.0 document as XML, such as the one of a graph, see `bpmn_document`.

    The document is written as it was read, including its diagram and extensions,
    so a document read and written again keeps everything that the schema binds.

    Examples:
        >>> g = from_bpmn(parse_bpmn('''
        ...     <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        ...       <process id="p">
        ...         <startEvent id="s"><timerEventDefinition><timeDuration>PT1M</timeDuration></timerEventDefinition></startEvent>
        ...         <subProcess id="sub"><task id="t" name="Check order" agent="checker"/></subProcess>
        ...         <sequenceFlow id="f" sourceRef="s" targetRef="sub">
        ...           <conditionExpression language="python">ok</conditionExpression>
        ...         </sequenceFlow>
        ...       </process>
        ...     </definitions>
        ... '''))
        >>> g.nodes["s"].value["event"], g.nodes["s"].value["timeDuration"], g.nodes["t"].value["parent"]
        ('timer', 'PT1M', 'sub')
        >>> print(render_bpmn(bpmn_document(g)), end="")
        <?xml version="1.0" encoding="UTF-8"?>
        <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
          <process id="p">
            <startEvent id="s">
              <timerEventDefinition>
                <timeDuration>PT1M</timeDuration>
              </timerEventDefinition>
            </startEvent>
            <subProcess id="sub">
              <task id="t" agent="checker" name="Check order"/>
            </subProcess>
            <sequenceFlow id="f" sourceRef="s" targetRef="sub">
              <conditionExpression language="python">ok</conditionExpression>
            </sequenceFlow>
          </process>
        </definitions>
        >>> from_bpmn(parse_bpmn(render_bpmn(bpmn_document(g)))) == g
        True
    """
    return _SERIALIZER.render(
        document, ns_map={None: BPMN_NAMESPACE, "xsi": Namespace.XSI.uri}
    )


def bpmn_problems(
    graph: BpmnGraph,
    *,
    supported: tuple[type[bpmn.TBaseElement], ...] = BPMN_NODES,
    events: Mapping[
        type[bpmn.TBaseElement], tuple[type[bpmn.TBaseElement] | None, ...]
    ] = {},
) -> list[str]:
    """Reports the structural problems of a process.

    Args:
        graph: The process.
        supported: The generated classes of the elements that are allowed, with their subclasses.
        events: The allowed event definitions of event classes, `None` for plain events,
            event classes without an entry may have any.

    Returns:
        A description of every problem, empty if there is none.

    Examples:
        >>> graph = from_bpmn(parse_bpmn('''
        ...     <definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
        ...       <process id="p"><task id="t"/><complexGateway id="g"/></process>
        ...     </definitions>
        ... '''))
        >>> bpmn_problems(graph, supported=(bpmn.TTask, bpmn.TStartEvent))
        ['The process has no start event', "Unsupported element complexGateway 'g'"]
    """
    problems: list[str] = []

    if not any(
        issubclass(bpmn_class(node.value["type"]), bpmn.TStartEvent)
        and node.value.get("parent") is None
        for node in graph.nodes.values()
    ):
        problems.append("The process has no start event")

    for key, node in graph.nodes.items():
        kind, cls = node.value["type"], bpmn_class(node.value["type"])
        event = bpmn_event_class(node.value)
        allowed = next(
            (allowed for base, allowed in events.items() if issubclass(cls, base)),
            None,
        )

        if not issubclass(cls, supported):
            problems.append(f"Unsupported element {kind} {key!r}")
        elif allowed is not None and not any(
            event is definition
            or (
                event is not None
                and definition is not None
                and issubclass(event, definition)
            )
            for definition in allowed
        ):
            problems.append(
                f"Unsupported {node.value.get('event')} event definition of {kind} {key!r}"
            )

        if (
            issubclass(cls, bpmn.TBoundaryEvent)
            and node.value.get("attachedToRef") not in graph.nodes
        ):
            problems.append(f"Boundary event {key!r} is attached to no activity")

    problems.extend(
        f"Event-based gateway {edge.source.key!r} leads to {edge.target.key!r}"
        for edge in graph.edges.values()
        if issubclass(bpmn_class(edge.source.value["type"]), bpmn.TEventBasedGateway)
        and issubclass(bpmn_class(edge.value["type"]), bpmn.TSequenceFlow)
        and not issubclass(
            bpmn_class(edge.target.value["type"]),
            bpmn.TReceiveTask | bpmn.TIntermediateCatchEvent,
        )
    )

    return problems
