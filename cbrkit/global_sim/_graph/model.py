from collections.abc import Hashable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

NodeKey = TypeVar("NodeKey")
NodeData = TypeVar("NodeData")
EdgeKey = TypeVar("EdgeKey")
EdgeData = TypeVar("EdgeData")
GraphData = TypeVar("GraphData")


class EdgeProtocol(Hashable, Protocol[EdgeData, NodeKey]):
    source: NodeKey
    target: NodeKey
    data: EdgeData


class NodeProtocol(Hashable, Protocol[NodeData]):
    data: NodeData


@dataclass
class Graph(Generic[GraphData, NodeKey, NodeData, EdgeKey, EdgeData]):
    nodes: dict[NodeKey, NodeProtocol[NodeData]]
    edges: dict[EdgeKey, EdgeProtocol[EdgeData, NodeKey]]
    data: GraphData
