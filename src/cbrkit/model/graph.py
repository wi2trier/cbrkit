from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from frozendict import frozendict
from pydantic import BaseModel

from ..helpers import identity, optional_dependencies
from ..typing import ConversionFunc, StructuredValue

__all__ = [
    "Node",
    "Edge",
    "Graph",
    "GraphElementType",
    "SerializedEdge",
    "SerializedGraph",
    "to_dict",
    "from_dict",
    "is_sequential",
    "from_rustworkx",
    "to_rustworkx",
    "to_rustworkx_with_lookup",
    "from_networkx",
    "to_networkx",
    "NetworkxNode",
    "NetworkxEdge",
    "to_sequence",
    "to_graphviz",
]

type GraphElementType = Literal["node", "edge"]


class SerializedEdge[K, E](BaseModel):
    source: K
    target: K
    value: E


class SerializedGraph[K, N, E, G](BaseModel):
    nodes: Mapping[K, N]
    edges: Mapping[K, SerializedEdge[K, E]]
    value: G


@dataclass(slots=True, frozen=True)
class Node[K, N](StructuredValue[N]):
    key: K

    def dump(self, converter: ConversionFunc[N, N] = identity) -> N:
        return converter(self.value)

    @classmethod
    def load(
        cls,
        key: K,
        obj: N,
        converter: ConversionFunc[N, N] = identity,
    ) -> Node[K, N]:
        return cls(
            converter(obj),
            key,
        )


@dataclass(slots=True, frozen=True)
class Edge[K, N, E](StructuredValue[E]):
    key: K
    source: Node[K, N]
    target: Node[K, N]

    def dump(self, converter: ConversionFunc[E, E] = identity) -> SerializedEdge[K, E]:
        return SerializedEdge(
            source=self.source.key,
            target=self.target.key,
            value=converter(self.value),
        )

    @classmethod
    def load(
        cls,
        key: K,
        obj: SerializedEdge[K, E],
        nodes: Mapping[K, Node[K, N]],
        converter: ConversionFunc[E, E] = identity,
    ) -> Edge[K, N, E]:
        return cls(
            converter(obj.value),
            key,
            nodes[obj.source],
            nodes[obj.target],
        )


@dataclass(slots=True, frozen=True)
class Graph[K, N, E, G](StructuredValue[G]):
    nodes: frozendict[K, Node[K, N]]
    edges: frozendict[K, Edge[K, N, E]]

    def dump(
        self,
        node_converter: ConversionFunc[N, N] = identity,
        edge_converter: ConversionFunc[E, E] = identity,
        graph_converter: ConversionFunc[G, G] = identity,
    ) -> SerializedGraph[K, N, E, G]:
        return SerializedGraph(
            nodes={key: node.dump(node_converter) for key, node in self.nodes.items()},
            edges={key: edge.dump(edge_converter) for key, edge in self.edges.items()},
            value=graph_converter(self.value),
        )

    @classmethod
    def load(
        cls,
        g: SerializedGraph[K, N, E, G],
        node_converter: ConversionFunc[N, N] = identity,
        edge_converter: ConversionFunc[E, E] = identity,
        graph_converter: ConversionFunc[G, G] = identity,
    ) -> Graph[K, N, E, G]:
        nodes = frozendict(
            (key, Node.load(key, value, node_converter))
            for key, value in g.nodes.items()
        )
        edges = frozendict(
            (key, Edge.load(key, value, nodes, edge_converter))
            for key, value in g.edges.items()
        )
        return cls(graph_converter(g.value), nodes, edges)

    @classmethod
    def build(
        cls,
        nodes: Iterable[Node[K, N]],
        edges: Iterable[Edge[K, N, E]],
        value: G,
    ) -> Graph[K, N, E, G]:
        node_map = frozendict((node.key, node) for node in nodes)
        edge_map = frozendict((edge.key, edge) for edge in edges)

        return cls(value, node_map, edge_map)


def to_dict[K, N, E, G](
    g: Graph[K, N, E, G],
    node_converter: ConversionFunc[N, N] = identity,
    edge_converter: ConversionFunc[E, E] = identity,
    graph_converter: ConversionFunc[G, G] = identity,
) -> Mapping[str, Any]:
    return g.dump(node_converter, edge_converter, graph_converter).model_dump()


def from_dict(
    g: Any,
    node_converter: ConversionFunc[Any, Any] = identity,
    edge_converter: ConversionFunc[Any, Any] = identity,
    graph_converter: ConversionFunc[Any, Any] = identity,
) -> Graph[Any, Any, Any, Any]:
    return Graph.load(
        SerializedGraph.model_validate(g),
        node_converter,
        edge_converter,
        graph_converter,
    )


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
        new_ids = ng.add_nodes_from(node.value for node in g.nodes.values())
        id_map = {
            old_id: new_id
            for old_id, new_id in zip(g.nodes.keys(), new_ids, strict=True)
        }
        ng.add_edges_from(
            (
                id_map[edge.source.key],
                id_map[edge.target.key],
                edge.value,
            )
            for edge in g.edges.values()
        )

        return ng, {new_id: old_id for old_id, new_id in id_map.items()}

    def to_rustworkx[N, E](g: Graph[Any, N, E, Any]) -> rustworkx.PyDiGraph[N, E]:
        return to_rustworkx_with_lookup(g)[0]

    def from_rustworkx[N, E](g: rustworkx.PyDiGraph[N, E]) -> Graph[int, N, E, Any]:
        nodes = frozendict(
            (idx, Node(key=idx, value=g.get_node_data(idx))) for idx in g.node_indices()
        )
        edges = frozendict(
            (
                edge_id,
                Edge(
                    key=edge_id,
                    source=nodes[source_id],
                    target=nodes[target_id],
                    value=edge_data,
                ),
            )
            for edge_id, (source_id, target_id, edge_data) in g.edge_index_map().items()
        )

        return Graph(nodes=nodes, edges=edges, value=g.attrs)


class NetworkxNode[K, N](TypedDict):
    key: K
    value: N
    obj: Node[K, N]


class NetworkxEdge[K, N, E](TypedDict):
    key: K
    value: E
    obj: Edge[K, N, E]


with optional_dependencies():
    import networkx as nx

    def to_networkx[K, N, E](g: Graph[K, N, E, Any]) -> nx.DiGraph:
        ng = nx.DiGraph()
        ng.graph = g.value

        ng.add_nodes_from(
            (
                node.key,
                NetworkxNode(
                    key=node.key,
                    value=node.value,
                    obj=node,
                ),
            )
            for node in g.nodes.values()
        )

        ng.add_edges_from(
            (
                edge.source.key,
                edge.target.key,
                NetworkxEdge(
                    key=edge.key,
                    value=edge.value,
                    obj=edge,
                ),
            )
            for edge in g.edges.values()
        )

        return ng

    def from_networkx(g: nx.DiGraph) -> Graph[Any, Any, Any, Any]:
        nodes = frozendict(
            (idx, Node(key=idx, value=data)) for idx, data in g.nodes(data=True)
        )

        edges = frozendict(
            (
                idx,
                Edge(
                    key=idx,
                    source=nodes[source_id],
                    target=nodes[target_id],
                    value=edge_data,
                ),
            )
            for idx, (source_id, target_id, edge_data) in enumerate(g.edges(data=True))
        )

        return Graph(nodes=nodes, edges=edges, value=g.graph)


with optional_dependencies():
    from pygraphviz import AGraph

    def to_graphviz[N, E, G](
        g: Graph[Any, N, E, G],
        name: str,
        strict: bool,
        directed: bool,
        node_converter: ConversionFunc[N, Mapping[str, str]] | None = None,
        edge_converter: ConversionFunc[E, Mapping[str, str]] | None = None,
        graph_converter: ConversionFunc[G, Mapping[str, str]] | None = None,
        node_attr: Mapping[str, str] | None = None,
        edge_attr: Mapping[str, str] | None = None,
        graph_attr: Mapping[str, str] | None = None,
    ) -> AGraph:
        gv = AGraph(
            name=name,
            strict=strict,
            directed=directed,
        )
        gv.graph_attr.update(graph_attr or {})
        gv.node_attr.update(node_attr or {})
        gv.edge_attr.update(edge_attr or {})

        if node_converter is not None:
            for node in g.nodes.values():
                gv.add_node(node.key, **node_converter(node.value))

        if edge_converter is not None:
            for edge in g.edges.values():
                gv.add_edge(
                    edge.source.key,
                    edge.target.key,
                    key=edge.key,
                    **edge_converter(edge.value),
                )

        if graph_converter is not None:
            gv.graph_attr.update(graph_converter(g.value))

        return gv
