from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

import immutables

from ...helpers import optional_dependencies
from ...typing import StructuredValue


@dataclass(slots=True, frozen=True)
class GraphSim[K](StructuredValue[float]):
    value: float
    node_mappings: dict[K, K]
    edge_mappings: dict[K, K]


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
class Node[K, N](StructuredValue[N]):
    key: K
    value: N

    def to_dict(self) -> SerializedNode[N]:
        return {"data": self.value}

    @classmethod
    def from_dict(
        cls,
        key: K,
        data: SerializedNode[N],
    ) -> Node[K, N]:
        return cls(key, data["data"])


@dataclass(slots=True, frozen=True)
class Edge[K, N, E](StructuredValue[E]):
    key: K
    source: Node[K, N]
    target: Node[K, N]
    value: E

    def to_dict(self) -> SerializedEdge[K, E]:
        return {
            "source": self.source.key,
            "target": self.target.key,
            "data": self.value,
        }

    @classmethod
    def from_dict(
        cls,
        key: K,
        data: SerializedEdge[K, E],
        nodes: Mapping[K, Node[K, N]],
    ) -> Edge[K, N, E]:
        return cls(
            key,
            nodes[data["source"]],
            nodes[data["target"]],
            data["data"],
        )


@dataclass(slots=True, frozen=True)
class Graph[K, N, E, G](StructuredValue[G]):
    nodes: immutables.Map[K, Node[K, N]]
    edges: immutables.Map[K, Edge[K, N, E]]
    value: G

    def to_dict(self) -> SerializedGraph[K, N, E, G]:
        return {
            "nodes": {key: node.to_dict() for key, node in self.nodes.items()},
            "edges": {key: edge.to_dict() for key, edge in self.edges.items()},
            "data": self.value,
        }

    @classmethod
    def from_dict(
        cls,
        g: SerializedGraph[K, N, E, G],
    ) -> Graph[K, N, E, G]:
        nodes = immutables.Map(
            (key, Node.from_dict(key, value)) for key, value in g["nodes"].items()
        )
        edges = immutables.Map(
            (key, Edge.from_dict(key, value, nodes))
            for key, value in g["edges"].items()
        )
        return cls(nodes, edges, g["data"])

    @classmethod
    def build(
        cls,
        nodes: Iterable[Node[K, N]],
        edges: Iterable[Edge[K, N, E]],
        data: G,
    ) -> Graph[K, N, E, G]:
        node_map = immutables.Map((node.key, node) for node in nodes)
        edge_map = immutables.Map((edge.key, edge) for edge in edges)

        return cls(node_map, edge_map, data)


def to_dict[K, N, E, G](g: Graph[K, N, E, G]) -> SerializedGraph[K, N, E, G]:
    return g.to_dict()


def from_dict[K, N, E, G](g: SerializedGraph[K, N, E, G]) -> Graph[K, N, E, G]:
    return Graph.from_dict(g)


def is_sequential[K, N, E, G](g: Graph[K, N, E, G]) -> bool:
    """
    Check if a graph is a sequential workflow.

    A sequential workflow is defined as a directed graph where:
    - Each node (except the last) has exactly one outgoing edge
    - Each node (except the first) has exactly one incoming edge
    - The graph forms a single path with no cycles or branches

    Args:
        g: The graph to check

    Returns:
        True if the graph is a sequential workflow, False otherwise
    """

    if not g.nodes:
        return True

    # Count incoming and outgoing edges for each node
    in_degree = {node.key: 0 for node in g.nodes.values()}
    out_degree = {node.key: 0 for node in g.nodes.values()}

    for edge in g.edges.values():
        in_degree[edge.target.key] += 1
        out_degree[edge.source.key] += 1

    # Check degrees match sequential pattern
    start_nodes = [k for k, count in in_degree.items() if count == 0]
    end_nodes = [k for k, count in out_degree.items() if count == 0]

    # Must have exactly one start and one end
    if len(start_nodes) != 1 or len(end_nodes) != 1:
        return False

    # All other nodes must have exactly one in and one out edge
    for node_key in g.nodes.keys():
        if node_key not in start_nodes and node_key not in end_nodes:
            if in_degree[node_key] != 1 or out_degree[node_key] != 1:
                return False

    return True


with optional_dependencies():
    import rustworkx

    def to_rustworkx_with_lookup[K, N, E](
        g: Graph[K, N, E, Any],
    ) -> tuple[rustworkx.PyDiGraph[N, E], dict[int, K]]:
        ng = rustworkx.PyDiGraph(attrs=g.value)
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
                    edge.value,
                )
                for edge in g.edges.values()
            ]
        )

        return ng, {new_id: old_id for old_id, new_id in id_map.items()}

    def to_rustworkx[N, E](g: Graph[Any, N, E, Any]) -> rustworkx.PyDiGraph[N, E]:
        return to_rustworkx_with_lookup(g)[0]

    def from_rustworkx[N, E](g: rustworkx.PyDiGraph[N, E]) -> Graph[int, N, E, Any]:
        nodes = immutables.Map(
            (idx, Node(idx, g.get_node_data(idx))) for idx in g.node_indices()
        )
        edges = immutables.Map(
            (edge_id, Edge(edge_id, nodes[source_id], nodes[target_id], edge_data))
            for edge_id, (source_id, target_id, edge_data) in g.edge_index_map().items()
        )

        return Graph(nodes, edges, g.attrs)


with optional_dependencies():
    import networkx as nx

    def to_networkx[K, N, E](g: Graph[K, N, E, Any]) -> nx.DiGraph:
        ng = nx.DiGraph()
        ng.graph = g.value

        ng.add_nodes_from(
            (
                node.key,
                (
                    node.value
                    if isinstance(node.value, Mapping)
                    else {"data": node.value}
                ),
            )
            for node in g.nodes.values()
        )

        ng.add_edges_from(
            (
                edge.source.key,
                edge.target.key,
                (
                    {**edge.value, "key": edge.key}
                    if isinstance(edge.value, Mapping)
                    else {"data": edge.value, "key": edge.key}
                ),
            )
            for edge in g.edges.values()
        )

        return ng

    def from_networkx(g: nx.DiGraph) -> Graph[Any, Any, Any, Any]:
        nodes = immutables.Map(
            (idx, Node(idx, data)) for idx, data in g.nodes(data=True)
        )

        edges = immutables.Map(
            (idx, Edge(idx, nodes[source_id], nodes[target_id], edge_data))
            for idx, (source_id, target_id, edge_data) in enumerate(g.edges(data=True))
        )

        return Graph(nodes, edges, g.graph)
