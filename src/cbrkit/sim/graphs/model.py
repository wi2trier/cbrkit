from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import immutables
from pydantic import BaseModel

from ...helpers import optional_dependencies
from ...typing import StructuredValue

type ElementType = Literal["node", "edge"]


@dataclass(slots=True, frozen=True)
class GraphSim[K](StructuredValue[float]):
    value: float
    node_mapping: dict[K, K]
    edge_mapping: dict[K, K]
    node_similarities: dict[K, float]
    edge_similarities: dict[K, float]


class ElementMatcher[T](Protocol):
    def __call__(self, x: T, y: T, /) -> bool: ...


def default_element_matcher(x: Any, y: Any) -> bool:
    return type(x) is type(y)


class SerializedNode[N](BaseModel):
    value: N


class SerializedEdge[K, E](BaseModel):
    source: K
    target: K
    value: E


class SerializedGraph[K, N, E, G](BaseModel):
    nodes: Mapping[K, SerializedNode[N]]
    edges: Mapping[K, SerializedEdge[K, E]]
    value: G


@dataclass(slots=True, frozen=True)
class Node[K, N](StructuredValue[N]):
    key: K
    value: N

    def dump(self) -> SerializedNode[N]:
        return SerializedNode(value=self.value)

    @classmethod
    def load(
        cls,
        key: K,
        obj: SerializedNode[N],
    ) -> Node[K, N]:
        return cls(key, obj.value)


@dataclass(slots=True, frozen=True)
class Edge[K, N, E](StructuredValue[E]):
    key: K
    source: Node[K, N]
    target: Node[K, N]
    value: E

    def dump(self) -> SerializedEdge[K, E]:
        return SerializedEdge(
            source=self.source.key,
            target=self.target.key,
            value=self.value,
        )

    @classmethod
    def load(
        cls,
        key: K,
        obj: SerializedEdge[K, E],
        nodes: Mapping[K, Node[K, N]],
    ) -> Edge[K, N, E]:
        return cls(
            key,
            nodes[obj.source],
            nodes[obj.target],
            obj.value,
        )


@dataclass(slots=True, frozen=True)
class Graph[K, N, E, G](StructuredValue[G]):
    nodes: immutables.Map[K, Node[K, N]]
    edges: immutables.Map[K, Edge[K, N, E]]
    value: G

    def dump(self) -> SerializedGraph[K, N, E, G]:
        return SerializedGraph(
            nodes={key: node.dump() for key, node in self.nodes.items()},
            edges={key: edge.dump() for key, edge in self.edges.items()},
            value=self.value,
        )

    @classmethod
    def load(
        cls,
        g: SerializedGraph[K, N, E, G],
    ) -> Graph[K, N, E, G]:
        nodes = immutables.Map(
            (key, Node.load(key, value)) for key, value in g.nodes.items()
        )
        edges = immutables.Map(
            (key, Edge.load(key, value, nodes)) for key, value in g.edges.items()
        )
        return cls(nodes, edges, g.value)

    @classmethod
    def build(
        cls,
        nodes: Iterable[Node[K, N]],
        edges: Iterable[Edge[K, N, E]],
        value: G,
    ) -> Graph[K, N, E, G]:
        node_map = immutables.Map((node.key, node) for node in nodes)
        edge_map = immutables.Map((edge.key, edge) for edge in edges)

        return cls(node_map, edge_map, value)


def to_dict[K, N, E, G](g: Graph[K, N, E, G]) -> Mapping[str, Any]:
    return g.dump().model_dump()


def from_dict(g: Any) -> Graph[Any, Any, Any, Any]:
    return Graph.load(SerializedGraph.model_validate(g))


def load[K](data: Mapping[K, Any]) -> dict[K, Graph[Any, Any, Any, Any]]:
    return {key: from_dict(value) for key, value in data.items()}


def dump[T, K, N, E, G](
    data: Mapping[T, Graph[K, N, E, G]],
) -> dict[T, SerializedGraph[K, N, E, G]]:
    return {key: value.dump() for key, value in data.items()}


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


def to_sequence[K, N, E, G](
    graph: Graph[K, N, E, G],
) -> tuple[list[Node[K, N]], list[Edge[K, N, E]]]:
    """
    Extract nodes and edges of a graph in sequential order.

    Args:
        graph: The graph to extract from.

    Returns:
        A tuple containing a list of nodes and a list of edges in sequential order.
    """

    in_degree = {node.key: 0 for node in graph.nodes.values()}

    for edge in graph.edges.values():
        in_degree[edge.target.key] += 1

    start_nodes = [node for node in graph.nodes.values() if in_degree[node.key] == 0]

    if len(start_nodes) != 1:
        raise ValueError("Graph does not have a unique start node")

    start_node = start_nodes[0]

    nodes: list[Node[K, N]] = []
    edges: list[Edge[K, N, E]] = []
    current_node = start_node
    visited_nodes = set()

    while current_node and current_node.key not in visited_nodes:
        nodes.append(current_node)
        visited_nodes.add(current_node.key)
        outgoing_edges = [
            edge for edge in graph.edges.values() if edge.source.key == current_node.key
        ]

        if len(outgoing_edges) > 1:
            raise ValueError(
                "Graph is not sequential (node has multiple outgoing edges)"
            )

        if outgoing_edges:
            edges.append(outgoing_edges[0])
            current_node = outgoing_edges[0].target
        else:
            current_node = None

    return nodes, edges


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
