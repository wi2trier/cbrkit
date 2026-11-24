"""Conversions between graphs and networkx multigraphs."""

from collections.abc import Hashable
from typing import Any

import networkx as nx

from ._graph import (
    Graph,
    NetworkxEdge,
    NetworkxNode,
    SerializedEdge,
    SerializedGraph,
    _load,
)

__all__ = ["from_networkx", "to_networkx"]

# aliases are evaluated lazily, since the classes are generic only in the stubs
type _DiGraph = nx.DiGraph[Any]
type _MultiDiGraph = nx.MultiDiGraph[Any]


def to_networkx[K: Hashable, N, E](g: Graph[K, N, E, Any]) -> _MultiDiGraph:
    """Convert a graph to a networkx MultiDiGraph.

    A multigraph is used because `Graph` allows parallel edges between the same
    pair of nodes, which a plain `DiGraph` would silently collapse into one.
    """
    ng = nx.MultiDiGraph()
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
            dict(
                NetworkxEdge(
                    key=edge.key,
                    value=edge.value,
                    obj=edge,
                )
            ),
        )
        for edge in g.edges.values()
    )

    return ng


def from_networkx(g: _DiGraph) -> Graph[Any, Any, Any, Any]:
    """Construct a graph from a networkx DiGraph or MultiDiGraph.

    Values are read from the `value` attribute and edge keys from the `key`
    attribute that `to_networkx` writes, so the conversion round-trips.
    Without them, the whole attribute dict is the value and edges are keyed
    by position.

    Examples:
        >>> from cbrkit.model.graph import from_dict
        >>> g = from_dict({
        ...     "nodes": {"a": 1, "b": 2},
        ...     "edges": {"e1": {"source": "a", "target": "b", "value": 3}},
        ... })
        >>> from_networkx(to_networkx(g)) == g
        True
    """
    return _load(
        SerializedGraph(
            nodes={key: data.get("value", data) for key, data in g.nodes(data=True)},
            edges={
                data.get("key", idx): SerializedEdge(
                    source=source, target=target, value=data.get("value", data)
                )
                for idx, (source, target, data) in enumerate(g.edges(data=True))
            },
            value=g.graph,
        )
    )
