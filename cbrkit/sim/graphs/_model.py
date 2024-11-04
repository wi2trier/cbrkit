from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import immutables

__all__ = [
    "Node",
    "Edge",
    "Graph",
    "SerializedNode",
    "SerializedEdge",
    "SerializedGraph",
    "to_dict",
    "from_dict",
]


class HasData[T](Protocol):
    data: T


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
class Node[K, N](HasData[N]):
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
class Edge[K, N, E](HasData[E]):
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
class Graph[K, N, E, G](HasData[G]):
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
        g: SerializedGraph[K, N, E, G],
    ) -> "Graph[K, N, E, G]":
        nodes = immutables.Map(
            (key, Node.from_dict(key, value)) for key, value in g["nodes"].items()
        )
        edges = immutables.Map(
            (key, Edge.from_dict(key, value, nodes))
            for key, value in g["edges"].items()
        )
        return cls(nodes, edges, g["data"])


def to_dict[K, N, E, G](g: Graph[K, N, E, G]) -> SerializedGraph[K, N, E, G]:
    return g.to_dict()


def from_dict[K, N, E, G](g: SerializedGraph[K, N, E, G]) -> Graph[K, N, E, G]:
    return Graph.from_dict(g)


try:
    import rustworkx

    def to_rustworkx[N, E](g: Graph[Any, N, E, Any]) -> "rustworkx.PyDiGraph[N, E]":
        ng = rustworkx.PyDiGraph(attrs=g.data)
        new_ids = ng.add_nodes_from(list(g.nodes.values()))
        id_map = {
            old_id: new_id
            for old_id, new_id in zip(g.nodes.keys(), new_ids, strict=True)
        }
        ng.add_edges_from(
            [
                (
                    id_map[edge.source.key],
                    id_map[edge.target.key],
                    edge.data,
                )
                for edge in g.edges.values()
            ]
        )

        return ng

    def from_rustworkx[N, E](g: "rustworkx.PyDiGraph[N, E]") -> Graph[int, N, E, Any]:
        nodes = immutables.Map(
            (idx, Node(idx, g.get_node_data(idx))) for idx in g.node_indices()
        )
        edges = immutables.Map(
            (edge_id, Edge(edge_id, nodes[source_id], nodes[target_id], edge_data))
            for edge_id, (
                source_id,
                target_id,
                edge_data,
            ) in g.edge_index_map().items()
        )

        return Graph(nodes, edges, g.attrs)

    __all__ += ["to_rustworkx", "from_rustworkx"]

except ImportError:
    pass

try:
    import networkx as nx


    def to_networkx[N, E](g: Graph[Any, N, E, Any]) -> "nx.DiGraph":
        ng = nx.DiGraph()
        # Set graph attributes
        ng.graph.update(g.data)

        # Add nodes with their data
        for node in g.nodes.values():
            ng.add_node(node.key, data=node.data)

        # Add edges with their data
        for edge in g.edges.values():
            ng.add_edge(edge.source.key, edge.target.key,
                    key=edge.key, data=edge.data)

        return ng

    def from_networkx[N, E](g: "nx.DiGraph") -> Graph[Any, N, E, Any]:
        # Create nodes
        nodes = immutables.Map(
            (node_id, Node(node_id, g.nodes[node_id].get('data')))
            for node_id in g.nodes
            )

        # Create edges
        edges = immutables.Map(
            (edge_data.get('key', idx),
            Edge(
            edge_data.get('key', idx),
            nodes[source],
            nodes[target],
            edge_data.get('data')
            ))
            for idx, (source, target, edge_data) in enumerate(g.edges(data=True))
        )

        return Graph(nodes, edges, dict(g.graph))

    __all__ += ["to_networkx", "from_networkx"]

except ImportError:
    pass