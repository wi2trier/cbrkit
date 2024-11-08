from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict, cast, override

import immutables

from cbrkit.helpers import SimWrapper
from cbrkit.typing import (
    AnnotatedFloat,
    Float,
    SimPairFunc,
    SimSeqFunc,
)

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


@dataclass(slots=True, frozen=True)
class GraphSim[K](AnnotatedFloat):
    value: float
    node_mappings: dict[K, K]
    edge_mappings: dict[K, K]


class DataSimWrapper[V: HasData[Any], S: Float](SimWrapper, SimSeqFunc[V, S]):
    @override
    def __call__(self, pairs: Sequence[tuple[V, V]]) -> Sequence[S]:
        if self.kind == "pair":
            func = cast(SimPairFunc[V, S], self.func)
            return [func(x.data, y.data) for (x, y) in pairs]

        func = cast(SimSeqFunc[V, S], self.func)
        return func([(x.data, y.data) for x, y in pairs])


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
    ) -> Node[K, N]:
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
    ) -> Edge[K, N, E]:
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
    ) -> Graph[K, N, E, G]:
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

    def to_rustworkx_with_lookup[K, N, E](
        g: Graph[K, N, E, Any],
    ) -> tuple[rustworkx.PyDiGraph[N, E], dict[int, K]]:
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

    __all__ += ["to_rustworkx", "from_rustworkx"]

except ImportError:
    pass

try:
    import networkx as nx

    def to_networkx(g: Graph) -> nx.DiGraph:
        ng = nx.DiGraph()
        ng.graph = g.data

        ng.add_nodes_from(
            (
                node.key,
                (node.data if isinstance(node.data, Mapping) else {"data": node.data}),
            )
            for node in g.nodes
        )

        ng.add_edges_from(
            (
                edge.source.key,
                edge.target.key,
                (
                    {**edge.data, "key": edge.key}
                    if isinstance(edge.data, Mapping)
                    else {"data": edge.data, "key": edge.key}
                ),
            )
            for edge in g.edges.values()
        )

        return ng

    def from_networkx(g: nx.DiGraph) -> Graph:
        nodes = immutables.Map(
            (idx, Node(idx, data)) for idx, data in g.nodes(data=True)
        )

        edges = immutables.Map(
            (idx, Edge(idx, nodes[source_id], nodes[target_id], edge_data))
            for idx, (source_id, target_id, edge_data) in enumerate(g.edges(data=True))
        )

        return Graph(nodes, edges, g.graph)

    __all__ += ["to_networkx", "from_networkx"]

except ImportError:
    pass
