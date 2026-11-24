"""Conversions between graphs and rustworkx digraphs."""

from collections.abc import Hashable
from typing import Any

import rustworkx

from ...helpers import frozendict
from ._graph import Edge, Graph, Node

__all__ = ["from_rustworkx", "to_rustworkx", "to_rustworkx_with_lookup"]


def to_rustworkx_with_lookup[K: Hashable, N, E](
    g: Graph[K, N, E, Any],
) -> tuple[rustworkx.PyDiGraph[N, E], dict[int, K]]:
    """Convert a graph to a rustworkx digraph and return a node ID lookup."""
    ng = rustworkx.PyDiGraph(attrs=g.value)
    new_ids = ng.add_nodes_from(node.value for node in g.nodes.values())
    id_map = dict(zip(g.nodes.keys(), new_ids, strict=True))
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
    """Convert a graph to a rustworkx digraph."""
    return to_rustworkx_with_lookup(g)[0]


def from_rustworkx[N, E](g: rustworkx.PyDiGraph[N, E]) -> Graph[int, N, E, Any]:
    """Construct a graph from a rustworkx digraph."""
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
