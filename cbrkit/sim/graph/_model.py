from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, TypedDict, TypeVar

import immutables

from cbrkit.typing import KeyType

NodeData = TypeVar("NodeData")
EdgeData = TypeVar("EdgeData")
GraphData = TypeVar("GraphData")


class SerializedNode(TypedDict, Generic[NodeData]):
    data: NodeData


class SerializedEdge(TypedDict, Generic[KeyType, EdgeData]):
    source: KeyType
    target: KeyType
    data: EdgeData


class SerializedGraph(TypedDict, Generic[KeyType, NodeData, EdgeData, GraphData]):
    nodes: Mapping[KeyType, SerializedNode[NodeData]]
    edges: Mapping[KeyType, SerializedEdge[KeyType, EdgeData]]
    data: GraphData


@dataclass(slots=True, frozen=True)
class Node(Generic[KeyType, NodeData]):
    key: KeyType
    data: NodeData

    def to_dict(self) -> SerializedNode[NodeData]:
        return {"data": self.data}

    @classmethod
    def from_dict(
        cls,
        key: KeyType,
        data: SerializedNode[NodeData],
    ) -> "Node[KeyType, NodeData]":
        return cls(key, data["data"])


@dataclass(slots=True, frozen=True)
class Edge(Generic[KeyType, NodeData, EdgeData]):
    key: KeyType
    source: Node[KeyType, NodeData]
    target: Node[KeyType, NodeData]
    data: EdgeData

    def to_dict(self) -> SerializedEdge[KeyType, EdgeData]:
        return {
            "source": self.source.key,
            "target": self.target.key,
            "data": self.data,
        }

    @classmethod
    def from_dict(
        cls,
        key: KeyType,
        data: SerializedEdge[KeyType, EdgeData],
        nodes: Mapping[KeyType, Node[KeyType, NodeData]],
    ) -> "Edge[KeyType, NodeData, EdgeData]":
        return cls(
            key,
            nodes[data["source"]],
            nodes[data["target"]],
            data["data"],
        )


@dataclass(slots=True)
class Graph(Generic[KeyType, NodeData, EdgeData, GraphData]):
    nodes: immutables.Map[KeyType, Node[KeyType, NodeData]]
    edges: immutables.Map[KeyType, Edge[KeyType, NodeData, EdgeData]]
    data: GraphData

    def to_dict(self) -> SerializedGraph[KeyType, NodeData, EdgeData, GraphData]:
        return {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": {key: edge.to_dict() for key, edge in self.edges.items()},
            "data": self.data,
        }

    @classmethod
    def from_dict(
        cls,
        data: SerializedGraph[KeyType, NodeData, EdgeData, GraphData],
    ) -> "Graph[KeyType, NodeData, EdgeData, GraphData]":
        nodes = immutables.Map(
            (key, Node.from_dict(key, value)) for key, value in data["nodes"].items()
        )
        edges = immutables.Map(
            (key, Edge.from_dict(key, value, nodes))
            for key, value in data["edges"].items()
        )
        return cls(nodes, edges, data["data"])
