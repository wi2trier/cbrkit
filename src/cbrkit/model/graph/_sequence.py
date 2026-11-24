from collections.abc import Hashable

from ._graph import Edge, Graph, Node

__all__ = [
    "is_sequential",
    "to_sequence",
]


def is_sequential[K: Hashable, N, E, G](g: Graph[K, N, E, G]) -> bool:
    """Check if a graph is a sequential workflow, i.e. a single path.

    Every node has at most one incoming and one outgoing edge, and walking from the
    only node without an incoming edge visits all nodes.

    Examples:
        >>> from cbrkit.model.graph import from_dict
        >>> path = from_dict({"nodes": {"a": 1, "b": 2}, "edges": {"e": {"source": "a", "target": "b"}}})
        >>> is_sequential(path)
        True
        >>> path_and_cycle = from_dict({
        ...     "nodes": {"a": 1, "b": 2, "c": 3, "d": 4},
        ...     "edges": {
        ...         "e1": {"source": "a", "target": "b"},
        ...         "e2": {"source": "c", "target": "d"},
        ...         "e3": {"source": "d", "target": "c"},
        ...     },
        ... })
        >>> is_sequential(path_and_cycle)
        False
    """
    return not g.nodes or _walk(g) is not None


def _walk[K: Hashable, N, E, G](g: Graph[K, N, E, G]) -> list[Edge[K, N, E]] | None:
    """Returns the edges of a path graph in order, or None if it is no path."""
    outgoing: dict[K, Edge[K, N, E]] = {}
    targets: set[K] = set()

    for edge in g.edges.values():
        if edge.source.key in outgoing or edge.target.key in targets:
            return None

        outgoing[edge.source.key] = edge
        targets.add(edge.target.key)

    starts = [key for key in g.nodes if key not in targets]

    if len(starts) != 1:
        return None

    # In-degrees of at most one keep the walk from entering a cycle.
    edges: list[Edge[K, N, E]] = []
    current = starts[0]

    while current in outgoing:
        edges.append(outgoing[current])
        current = outgoing[current].target.key

    return edges if len(edges) == len(g.edges) == len(g.nodes) - 1 else None


def to_sequence[K: Hashable, N, E, G](
    graph: Graph[K, N, E, G],
) -> tuple[list[Node[K, N]], list[Edge[K, N, E]]]:
    """
    Extract nodes and edges of a sequential graph in order.

    Args:
        graph: The graph to extract from.

    Returns:
        A tuple containing a list of nodes and a list of edges in sequential order.

    Raises:
        ValueError: If the graph is not sequential.
    """
    if not graph.nodes:
        return [], []

    edges = _walk(graph)

    if edges is None:
        raise ValueError("Graph is not sequential")

    start = edges[0].source if edges else next(iter(graph.nodes.values()))

    return [start, *(edge.target for edge in edges)], edges
