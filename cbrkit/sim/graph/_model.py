from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import immutables


class SerializedNode[N](TypedDict):
    data: N


class SerializedEdge[K, E](TypedDict):
    source: K
    target: K
    data: E


class SerializedGraph[K, N, E, G](TypedDict):
    nodes: Mapping[K, SerializedNode[N]]
    edges: Mapping[K, SerializedEdge[K, E]]
    data: G


@dataclass(slots=True, frozen=True)
class Node[K, N]:
    key: K
    data: N

    def to_dict(self) -> SerializedNode[N]:
        return {"data": self.data}

    @classmethod
    def from_dict(
        cls,
        key: K,
        data: SerializedNode[N],
    ) -> "Node[K, N]":
        return cls(key, data["data"])


@dataclass(slots=True, frozen=True)
class Edge[K, N, E]:
    key: K
    source: Node[K, N]
    target: Node[K, N]
    data: E

    def to_dict(self) -> SerializedEdge[K, E]:
        return {
            "source": self.source.key,
            "target": self.target.key,
            "data": self.data,
        }

    @classmethod
    def from_dict(
        cls,
        key: K,
        data: SerializedEdge[K, E],
        nodes: Mapping[K, Node[K, N]],
    ) -> "Edge[K, N, E]":
        return cls(
            key,
            nodes[data["source"]],
            nodes[data["target"]],
            data["data"],
        )


@dataclass(slots=True, frozen=True)
class Graph[K, N, E, G]:
    nodes: immutables.Map[K, Node[K, N]]
    edges: immutables.Map[K, Edge[K, N, E]]
    data: G

    def to_dict(self) -> SerializedGraph[K, N, E, G]:
        return {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": {key: edge.to_dict() for key, edge in self.edges.items()},
            "data": self.data,
        }

    @classmethod
    def from_dict(
        cls,
        data: SerializedGraph[K, N, E, G],
    ) -> "Graph[K, N, E, G]":
        nodes = immutables.Map(
            (key, Node.from_dict(key, value)) for key, value in data["nodes"].items()
        )
        edges = immutables.Map(
            (key, Edge.from_dict(key, value, nodes))
            for key, value in data["edges"].items()
        )
        return cls(nodes, edges, data["data"])
