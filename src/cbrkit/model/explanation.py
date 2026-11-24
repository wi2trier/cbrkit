"""Serializable explanations of similarities for user interfaces, see `cbrkit.sim.explain`."""

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field

from ..schema import Model

__all__ = ["Explanation", "GraphExplanation"]


class Explanation(Model):
    """Breakdown of a similarity, by attribute for records and by mapping for graphs.

    Args:
        similarity: The similarity.
        query: The compared query value, or the key of a query node or edge.
        case: The compared case value, or the key of a case node or edge.
        weight: Relative importance within the parent attribute-value similarity.
        contribution: Share of the parent similarity, only defined for mean pooling.
        attributes: Breakdown of an attribute-value similarity by attribute.
        graph: Breakdown of a graph similarity by mapped element.
    """

    similarity: float
    query: Any = None
    case: Any = None
    weight: float | None = None
    contribution: float | None = None
    attributes: Mapping[str, Explanation] = Field(default_factory=dict)
    graph: GraphExplanation | None = None


class GraphExplanation(Model):
    """Mapping of a graph similarity in the order of the query elements.

    Args:
        exact: Whether the mapping is proven optimal.
        nodes: The mapped query nodes, with the node keys as query and case.
        edges: The mapped query edges, with the edge keys as query and case.
        unmapped_nodes: Keys of the query nodes without a counterpart.
        unmapped_edges: Keys of the query edges without a counterpart.
        node_coverage: Share of query nodes that are mapped.
        edge_coverage: Share of query edges that are mapped.
    """

    exact: bool
    nodes: Sequence[Explanation]
    edges: Sequence[Explanation]
    unmapped_nodes: Sequence[Any]
    unmapped_edges: Sequence[Any]
    node_coverage: float
    edge_coverage: float
