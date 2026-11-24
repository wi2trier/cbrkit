from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, ReadOnly, Self, TypedDict

from pydantic import BaseModel, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import CoreSchema, SchemaSerializer, core_schema

from ...helpers import frozendict
from ...typing import ConversionFunc

__all__ = [
    "Edge",
    "Graph",
    "GraphElementType",
    "NetworkxEdge",
    "NetworkxNode",
    "Node",
    "SerializedEdge",
    "SerializedGraph",
    "from_dict",
    "from_json",
    "to_dict",
]

type GraphElementType = Literal["node", "edge"]


def _unchanged(value: Any, /) -> Any:
    """Returns a value as it is, a non-generic identity, since type checkers solve a generic default once for all parameters."""
    return value


def _default_none(data: Any, *fields: str) -> Any:
    """Fills in None for missing value fields before validation."""
    if isinstance(data, Mapping):
        return {**dict.fromkeys(fields), **data}

    return data


class SerializedEdge[K: Hashable, E](BaseModel):
    """Serialized representation of a graph edge, whose value defaults to None."""

    source: K
    target: K
    value: E

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: Any) -> Any:
        return _default_none(data, "value")


class SerializedGraph[K: Hashable, N, E, G](BaseModel):
    """Serialized representation of a directed graph.

    The edges and the value default to empty and None, and every edge has to
    connect two of the nodes.
    `version` names the format, so graphs stored in files and databases stay readable:
    a graph without one is of version 1, and a later version reads the earlier ones.
    """

    version: Literal[1] = 1
    nodes: Mapping[K, N]
    edges: Mapping[K, SerializedEdge[K, E]] = Field(default_factory=dict)
    value: G

    @model_validator(mode="before")
    @classmethod
    def _defaults(cls, data: Any) -> Any:
        return _default_none(data, "value")

    @model_validator(mode="after")
    def _check_endpoints(self) -> Self:
        for key, edge in self.edges.items():
            for endpoint in (edge.source, edge.target):
                if endpoint not in self.nodes:
                    raise ValueError(
                        f"Edge {key!r} references unknown node {endpoint!r}"
                    )

        return self


@dataclass(slots=True, frozen=True, kw_only=True)
class Node[K: Hashable, N]:
    """A node in a graph with a key and typed value.

    Args:
        value: The value of the node.
        key: The key of the node in its graph.
    """

    value: N
    key: K


@dataclass(slots=True, frozen=True, kw_only=True)
class Edge[K: Hashable, N, E]:
    """A directed edge connecting two nodes in a graph.

    Args:
        value: The value of the edge.
        key: The key of the edge in its graph.
        source: The node the edge starts at.
        target: The node the edge ends at.
    """

    value: E
    key: K
    source: Node[K, N]
    target: Node[K, N]


@dataclass(slots=True, frozen=True, kw_only=True)
class Graph[K: Hashable, N, E, G]:
    """A directed graph with typed nodes, edges, and a graph-level value.

    Pydantic validates a graph from its serialized form, `SerializedGraph`, or a graph,
    and serializes it to that form in JSON mode, also in fields typed `Any`,
    while Python mode keeps the graph.
    Fields are annotated with the bare `Graph`,
    since pydantic treats a parametrized one as a plain dataclass.
    `from_dict` and `from_json` construct a graph with converters of its values, and `to_dict` serializes one.

    Args:
        nodes: The nodes by key.
        edges: The edges by key.
        value: The value of the graph itself.

    Examples:
        >>> from pydantic import TypeAdapter
        >>> adapter = TypeAdapter(Graph)
        >>> g = adapter.validate_json('{"nodes": {"a": 1}}')
        >>> g.nodes["a"].value, adapter.dump_python(g) is g
        (1, True)
        >>> adapter.dump_json(g)
        b'{"version":1,"nodes":{"a":1},"edges":{},"value":null}'
    """

    nodes: frozendict[K, Node[K, N]]
    edges: frozendict[K, Edge[K, N, E]]
    value: G
    __pydantic_serializer__: ClassVar[SchemaSerializer]

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        serialized = handler.generate_schema(SerializedGraph[Any, Any, Any, Any])
        loaded = core_schema.no_info_after_validator_function(_load, serialized)

        return core_schema.json_or_python_schema(
            json_schema=loaded,
            python_schema=core_schema.union_schema(
                [core_schema.is_instance_schema(cls), loaded]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                _dump, info_arg=False, return_schema=serialized, when_used="json"
            ),
        )

    @classmethod
    def build(
        cls,
        nodes: Iterable[Node[K, N]],
        edges: Iterable[Edge[K, N, E]],
        value: G,
    ) -> Self:
        """Construct a graph from iterables of nodes and edges, keyed by their keys."""
        return cls(
            nodes=frozendict((node.key, node) for node in nodes),
            edges=frozendict((edge.key, edge) for edge in edges),
            value=value,
        )


def _load[K: Hashable, N, E, G](
    g: SerializedGraph[K, Any, Any, Any],
    node_converter: ConversionFunc[Any, N] = _unchanged,
    edge_converter: ConversionFunc[Any, E] = _unchanged,
    graph_converter: ConversionFunc[Any, G] = _unchanged,
) -> Graph[K, N, E, G]:
    """The graph of a serialized graph, with its values converted."""
    nodes = frozendict(
        (key, Node(value=node_converter(value), key=key))
        for key, value in g.nodes.items()
    )
    edges = frozendict(
        (
            key,
            Edge(
                value=edge_converter(edge.value),
                key=key,
                source=nodes[edge.source],
                target=nodes[edge.target],
            ),
        )
        for key, edge in g.edges.items()
    )

    return Graph(nodes=nodes, edges=edges, value=graph_converter(g.value))


def _dump[K: Hashable, N, E, G](
    g: Graph[K, N, E, G],
    node_converter: ConversionFunc[N, Any] = _unchanged,
    edge_converter: ConversionFunc[E, Any] = _unchanged,
    graph_converter: ConversionFunc[G, Any] = _unchanged,
) -> SerializedGraph[K, Any, Any, Any]:
    """The serialized form of a graph, with its values converted."""
    return SerializedGraph(
        nodes={key: node_converter(node.value) for key, node in g.nodes.items()},
        edges={
            key: SerializedEdge(
                source=edge.source.key,
                target=edge.target.key,
                value=edge_converter(edge.value),
            )
            for key, edge in g.edges.items()
        },
        value=graph_converter(g.value),
    )


# pydantic serializes values of unknown type, such as cases in fields typed `Any`, with this serializer
Graph.__pydantic_serializer__ = SchemaSerializer(
    core_schema.is_instance_schema(
        Graph,
        serialization=core_schema.plain_serializer_function_ser_schema(
            _dump, info_arg=False, when_used="json"
        ),
    )
)


def to_dict[N, E, G](
    g: Graph[Any, N, E, G],
    *,
    node_converter: ConversionFunc[N, Any] = _unchanged,
    edge_converter: ConversionFunc[E, Any] = _unchanged,
    graph_converter: ConversionFunc[G, Any] = _unchanged,
) -> dict[str, Any]:
    """Convert a graph to a plain dictionary representation, with the converters of its values.

    Examples:
        >>> to_dict(from_dict({"nodes": {"a": 1}}), node_converter=str)
        {'version': 1, 'nodes': {'a': '1'}, 'edges': {}, 'value': None}
    """
    return _dump(g, node_converter, edge_converter, graph_converter).model_dump()


def from_dict[N, E, G](
    data: object,
    *,
    node_converter: ConversionFunc[Any, N] = _unchanged,
    edge_converter: ConversionFunc[Any, E] = _unchanged,
    graph_converter: ConversionFunc[Any, G] = _unchanged,
) -> Graph[Any, N, E, G]:
    """Construct a graph from a plain dictionary representation, validating it first, with the converters of its values.

    Raises:
        pydantic.ValidationError: If the data is no serialized graph, such as one with edges between unknown nodes.
    """
    return _load(
        SerializedGraph[Any, Any, Any, Any].model_validate(data),
        node_converter,
        edge_converter,
        graph_converter,
    )


def from_json[N, E, G](
    data: str | bytes,
    *,
    node_converter: ConversionFunc[Any, N] = _unchanged,
    edge_converter: ConversionFunc[Any, E] = _unchanged,
    graph_converter: ConversionFunc[Any, G] = _unchanged,
) -> Graph[str, N, E, G]:
    """Construct a graph from its JSON representation, validating it first, with the converters of its values.

    Raises:
        pydantic.ValidationError: If the data is no serialized graph, such as one with edges between unknown nodes.

    Examples:
        >>> g = from_json('{"nodes": {"a": 1, "b": 2}, "edges": {"e": {"source": "a", "target": "b"}}}')
        >>> g.edges["e"].source.key
        'a'
        >>> from_json('{"nodes": {}, "edges": {"e": {"source": "a", "target": "b"}}}')
        Traceback (most recent call last):
        pydantic_core._pydantic_core.ValidationError: ...
    """
    return _load(
        SerializedGraph[str, Any, Any, Any].model_validate_json(data),
        node_converter,
        edge_converter,
        graph_converter,
    )


class NetworkxNode[K: Hashable, N](TypedDict):
    """Typed dictionary representing a node in a networkx graph."""

    key: ReadOnly[K]
    value: ReadOnly[N]
    obj: ReadOnly[Node[K, N]]


class NetworkxEdge[K: Hashable, N, E](TypedDict):
    """Typed dictionary representing an edge in a networkx graph."""

    key: ReadOnly[K]
    value: ReadOnly[E]
    obj: ReadOnly[Edge[K, N, E]]
