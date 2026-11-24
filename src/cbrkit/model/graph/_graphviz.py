"""Conversion of graphs to Graphviz drawings."""

from collections.abc import Mapping
from typing import Any

# pygraphviz ships no type information
from pygraphviz import AGraph

from ...typing import ConversionFunc
from ._graph import Graph

__all__ = ["to_graphviz"]


def to_graphviz[N, E, G](
    g: Graph[Any, N, E, G],
    name: str = "",
    strict: bool = False,
    directed: bool = True,
    *,
    node_converter: ConversionFunc[N, Mapping[str, str]] | None = None,
    edge_converter: ConversionFunc[E, Mapping[str, str]] | None = None,
    graph_converter: ConversionFunc[G, Mapping[str, str]] | None = None,
    node_attr: Mapping[str, str] | None = None,
    edge_attr: Mapping[str, str] | None = None,
    graph_attr: Mapping[str, str] | None = None,
) -> AGraph:
    """Convert a graph to a pygraphviz AGraph for visualization.

    Every node and edge is drawn, the converters only add the Graphviz attributes of their values.

    Args:
        g: The graph.
        name: Name of the drawing.
        strict: Whether the drawing forbids parallel edges and self-loops.
        directed: Whether the edges are drawn as arrows.
        node_converter: Graphviz attributes of a node value.
        edge_converter: Graphviz attributes of an edge value.
        graph_converter: Graphviz attributes of the graph value.
        node_attr: Default attributes of all nodes.
        edge_attr: Default attributes of all edges.
        graph_attr: Attributes of the drawing.

    Examples:
        >>> from cbrkit.model.graph import from_dict
        >>> g = from_dict({"nodes": {"a": 1, "b": 2}, "edges": {"e": {"source": "a", "target": "b"}}})
        >>> gv = to_graphviz(g, node_converter=lambda value: {"label": str(value)})
        >>> gv.nodes(), gv.edges(), gv.get_node("a").attr["label"]
        (['a', 'b'], [('a', 'b')], '1')
    """
    gv = AGraph(name=name, strict=strict, directed=directed)
    gv.graph_attr.update(graph_attr or {})
    gv.node_attr.update(node_attr or {})
    gv.edge_attr.update(edge_attr or {})

    for node in g.nodes.values():
        gv.add_node(node.key, **(node_converter(node.value) if node_converter else {}))

    for edge in g.edges.values():
        gv.add_edge(
            edge.source.key,
            edge.target.key,
            key=edge.key,
            **(edge_converter(edge.value) if edge_converter else {}),
        )

    if graph_converter is not None:
        gv.graph_attr.update(graph_converter(g.value))

    return gv
