from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from ..helpers import batchify_sim, get_logger, getitem_or_getattr
from ..typing import (
    AggregatorFunc,
    AnySimFunc,
    BatchSimFunc,
    Float,
    SimSeq,
    StructuredValue,
)
from .aggregator import default_aggregator

__all__ = ["attribute_value", "AttributeValueSim"]

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class AttributeValueSim[S: Float](StructuredValue[float]):
    attributes: Mapping[str, S]


@dataclass(slots=True, frozen=True)
class attribute_value[V, S: Float](BatchSimFunc[V, AttributeValueSim[S]]):
    """Similarity function that computes the attribute value similarity between two cases.

    Args:
        attributes: A mapping of attribute names to the similarity functions to be used for those attributes.
        aggregator: A function that aggregates the local similarity scores for each attribute into a single global similarity.
        value_getter: A function that retrieves the value of an attribute from a case.
        default: The default similarity score to use when an error occurs during the computation of a similarity score.
            For example, if a case does not have an attribute that is required for the similarity computation.

    Examples:
        >>> equality = lambda x, y: 1.0 if x == y else 0.0
        >>> sim = attribute_value({
        ...     "name": equality,
        ...     "age": equality,
        ... })
        >>> scores = sim([
        ...     ({"name": "John", "age": 25}, {"name": "John", "age": 30}),
        ...     ({"name": "Jane", "age": 30}, {"name": "John", "age": 30}),
        ... ])
        >>> scores[0]
        AttributeValueSim(value=0.5, attributes={'name': 1.0, 'age': 0.0})
        >>> scores[1]
        AttributeValueSim(value=0.5, attributes={'name': 0.0, 'age': 1.0})
    """

    attributes: Mapping[str, AnySimFunc[Any, S]]
    aggregator: AggregatorFunc[str, S] = default_aggregator
    value_getter: Callable[[Any, str], Any] = getitem_or_getattr
    default: S | None = None

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[AttributeValueSim[S]]:
        if len(batches) == 0:
            return []

        local_sims: list[dict[str, S]] = [dict() for _ in range(len(batches))]

        for attr_name in self.attributes:
            logger.debug(f"Processing attribute {attr_name}")

            try:
                attribute_values = [
                    (self.value_getter(x, attr_name), self.value_getter(y, attr_name))
                    for x, y in batches
                ]
                sim_func = batchify_sim(self.attributes[attr_name])
                sim_func_result = sim_func(attribute_values)

                for idx, sim in enumerate(sim_func_result):
                    local_sims[idx][attr_name] = sim

            except Exception as e:
                if self.default is not None:
                    for idx in range(len(batches)):
                        local_sims[idx][attr_name] = self.default
                else:
                    raise e

        return [AttributeValueSim(self.aggregator(sims), sims) for sims in local_sims]
