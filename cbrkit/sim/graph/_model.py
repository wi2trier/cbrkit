from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

NodeKey = TypeVar("NodeKey")
NodeData = TypeVar("NodeData")
EdgeKey = TypeVar("EdgeKey")
EdgeData = TypeVar("EdgeData")
GraphData = TypeVar("GraphData")


@runtime_checkable
class EdgeProtocol(Hashable, Protocol[EdgeData, NodeKey]):
    source: NodeKey
    target: NodeKey
    data: EdgeData


@runtime_checkable
class NodeProtocol(Hashable, Protocol[NodeData]):
    data: NodeData


@dataclass(slots=True)
class Graph(Generic[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]):
    nodes: dict[NodeKey, NodeProtocol[NodeData]]
    edges: dict[EdgeKey, EdgeProtocol[EdgeData, NodeKey]]
    data: GraphData
