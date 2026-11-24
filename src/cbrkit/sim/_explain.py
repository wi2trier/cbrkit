"""Explanations of similarities for user interfaces."""

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

from ..model.explanation import Explanation
from ..model.similarity import AttributeValueSim
from ..typing import AnySimFunc, Float
from .aggregator import Aggregator
from .pooling import Mean

__all__ = ["Explainable", "explain"]


@runtime_checkable
class Explainable[V](Protocol):
    """A measure that breaks its similarities down for user interfaces, see `explain`.

    `cbrkit.sim.AttributeValue`, `cbrkit.sim.Combine`, and the graph measures implement it,
    and the wrappers such as `cbrkit.sim.Transpose` delegate to the measure they wrap.
    """

    def explain(self, case: V, query: V, similarity: Float, /) -> Explanation:
        """Explains the similarity of the case to the query, a flat explanation if the measure did not compute it."""
        ...


def explain[V](
    case: V, query: V, similarity: Float, measure: AnySimFunc[V, Any] | None = None
) -> Explanation:
    """Explains the similarity of a case to a query by the measure that computed it.

    An `Explainable` measure breaks it down, attribute-value similarities by attribute
    and graph similarities by the mapping of the query elements, both recursively,
    with the weights and, for mean pooling, the contributions of the attributes.
    Other measures explain the similarity by its value only.

    Args:
        case: The case.
        query: The query.
        similarity: The similarity of the case to the query.
        measure: The measure that computed the similarity.

    Examples:
        >>> from cbrkit.sim import Aggregator, AttributeValue
        >>> from cbrkit.sim.generic import Equality
        >>> measure = AttributeValue(
        ...     attributes={"a": Equality(), "b": Equality()},
        ...     aggregator=Aggregator(weights={"b": 3}),
        ... )
        >>> case, query = {"a": 1, "b": 3}, {"a": 1, "b": 2}
        >>> result = explain(case, query, measure([(case, query)])[0], measure)
        >>> result.similarity, {name: item.contribution for name, item in result.attributes.items()}
        (0.25, {'a': 0.25, 'b': 0.0})
        >>> from cbrkit.model.graph import from_dict
        >>> from cbrkit.sim.graphs import LAP
        >>> graph = from_dict({
        ...     "nodes": {"1": "A", "2": "B"},
        ...     "edges": {"e1": {"source": "1", "target": "2"}},
        ... })
        >>> query = from_dict({
        ...     "nodes": {"1": "A", "3": "C"},
        ...     "edges": {"e1": {"source": "1", "target": "3"}},
        ... })
        >>> sim = LAP(node_measure=Equality(), node_matcher=lambda case, query: case == query)
        >>> mapping = explain(graph, query, sim([(graph, query)])[0], sim).graph
        >>> mapping.unmapped_nodes, mapping.node_coverage
        (['3'], 0.5)
    """
    if isinstance(measure, Explainable):
        return measure.explain(case, query, similarity)

    return Explanation(similarity=float(similarity))


def explain_attributes(
    similarity: Float,
    aggregator: Aggregator[str],
    measures: Mapping[str, AnySimFunc[Any, Any]],
    values: Callable[[str], tuple[Any, Any]],
) -> Explanation:
    """Breaks an attribute-value similarity down by its named local similarities.

    Args:
        similarity: The similarity, a flat explanation unless it is an `AttributeValueSim`.
        aggregator: Aggregates the local similarities.
        measures: The measures of the local similarities by name.
        values: The case and query values that a local similarity compares by its name.
    """
    if not isinstance(similarity, AttributeValueSim):
        return Explanation(similarity=float(similarity))

    weights = {name: aggregator.weight(name) for name in similarity.similarities}
    total = sum(weight for weight in weights.values() if weight > 0)
    mean = isinstance(aggregator.pooling, Mean) and total > 0
    attributes: dict[str, Explanation] = {}

    for name, local in similarity.similarities.items():
        case, query = values(name)
        attributes[name] = explain(case, query, local, measures.get(name)).model_copy(
            update={
                "case": case,
                "query": query,
                "weight": weights[name],
                "contribution": weights[name] * float(local) / total if mean else None,
            }
        )

    return Explanation(similarity=float(similarity), attributes=attributes)
