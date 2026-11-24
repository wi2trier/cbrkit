"""Similarity functions that wrap others.

`Combine` and `AttributeTable` are measures that serialize by their kind.
`Transpose` and `Cache` need code, a conversion function or a runtime store,
so they are built in Python only.
"""

from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from .._util import get_value
from ..helpers import (
    batchify,
    frozendict,
    get_attribute,
    getitem_or_getattr,
    singleton,
)
from ..model import AttributeValueSim, Explanation
from ..schema import PythonOnly, serializable
from ..typing import (
    AnySimFunc,
    AttributeGetter,
    BatchSimFunc,
    ConversionFunc,
    Float,
    HasValue,
    SimSeq,
)
from ._explain import Explainable, explain, explain_attributes
from ._nested import NestedMeasure
from .aggregator import Aggregator, default_aggregator

__all__ = [
    "AttributeTable",
    "Cache",
    "Combine",
    "Transpose",
    "transpose_singleton",
    "transpose_value",
]


@dataclass(slots=True, frozen=True, kw_only=True)
class Transpose[V1, V2, S: Float](BatchSimFunc[V1, S], Explainable[V1]):
    """Transforms a similarity function from one type to another.

    Args:
        measure: The similarity function to be used on the converted values.
        conversion: A function that converts the input values from one type to another.

    Examples:
        >>> from cbrkit.sim.generic import Equality
        >>> sim = Transpose(
        ...     measure=Equality(),
        ...     conversion=lambda x: x.lower(),
        ... )
        >>> sim([("A", "a"), ("b", "B")])
        [1.0, 1.0]
    """

    measure: NestedMeasure[V2, S]
    conversion: ConversionFunc[V1, V2]

    @override
    def __call__(self, batches: Sequence[tuple[V1, V1]]) -> Sequence[S]:
        return batchify(self.measure)(
            [(self.conversion(x), self.conversion(y)) for x, y in batches]
        )

    @override
    def explain(self, case: V1, query: V1, similarity: Float) -> Explanation:
        return explain(
            self.conversion(case), self.conversion(query), similarity, self.measure
        )


def transpose_value[V, S: Float](
    func: AnySimFunc[V, S],
) -> BatchSimFunc[HasValue[V], S]:
    """Create a transposed similarity function that extracts values before comparing."""
    return Transpose(measure=func, conversion=get_value)


def transpose_singleton[V, S: Float](
    func: AnySimFunc[V, S],
) -> BatchSimFunc[tuple[V, ...], S]:
    """Create a transposed similarity function that unwraps singleton tuples."""
    return Transpose(measure=func, conversion=singleton)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Combine[V, S: Float](BatchSimFunc[V, AttributeValueSim[S]], Explainable[V]):
    """Combines several measures of the same values into one, keeping their similarities by name.

    Args:
        measures: The measures by name.
        aggregator: Aggregates the similarities of the measures by their names.

    Examples:
        >>> from cbrkit.sim.generic import Equality, Static
        >>> sim = Combine(measures={"equal": Equality(), "half": Static(value=0.5)})
        >>> [float(value) for value in sim([("a", "a"), ("a", "b")])]
        [0.75, 0.25]
    """

    measures: Mapping[str, NestedMeasure[V, S]]
    aggregator: Aggregator = default_aggregator
    kind: Literal["combine"] = field(default="combine", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[AttributeValueSim[S]]:
        named = {name: batchify(func)(batches) for name, func in self.measures.items()}
        rows = (
            {name: sims[idx] for name, sims in named.items()}
            for idx in range(len(batches))
        )

        return [
            AttributeValueSim(
                value=float(self.aggregator(row)), similarities=frozendict(row)
            )
            for row in rows
        ]

    @override
    def explain(self, case: V, query: V, similarity: Float) -> Explanation:
        # the measures compare the whole values
        return explain_attributes(
            similarity, self.aggregator, self.measures, lambda _: (case, query)
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class Cache[V, U, S: Float](BatchSimFunc[V, S], Explainable[V]):
    """Keeps the similarities of pairs in an explicit store to avoid computing them again.

    Args:
        measure: The similarity function to cache.
        store: The similarities by pair of cache keys, which the caller owns and may share or persist.
        conversion: Optional function to convert values to cache keys.

    Examples:
        >>> from cbrkit.sim.generic import Equality
        >>> store = {}
        >>> sim = Cache(measure=Equality(), store=store)
        >>> sim([("a", "a"), ("b", "b")])
        [1.0, 1.0]
        >>> sim([("a", "a")])
        [1.0]
        >>> len(store)
        2
    """

    measure: NestedMeasure[V, S]
    store: MutableMapping[tuple[U, U], S] = field(repr=False)
    conversion: ConversionFunc[V, U] | None = None

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[S]:
        transformed_batches = (
            [(self.conversion(x), self.conversion(y)) for x, y in batches]
            if self.conversion is not None
            else cast(list[tuple[U, U]], batches)
        )
        # the first index of every distinct pair that is not stored yet
        uncached: dict[tuple[U, U], int] = {}

        for idx, pair in enumerate(transformed_batches):
            if pair not in uncached and pair not in self.store:
                uncached[pair] = idx

        uncached_sims = batchify(self.measure)(
            [batches[idx] for idx in uncached.values()]
        )
        self.store.update(zip(uncached, uncached_sims, strict=True))

        return [self.store[pair] for pair in transformed_batches]

    @override
    def explain(self, case: V, query: V, similarity: Float) -> Explanation:
        return explain(case, query, similarity, self.measure)


_OTHER = object()


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AttributeTable[S: Float](BatchSimFunc[Any, S], Explainable[Any]):
    """Similarity measure chosen by a shared value of an attribute, such as the type of a node, or by a shared type.

    Pairs whose keys differ or are not listed get the `fallback` measure.
    Differing keys of a pair never select a measure, so `fallback` also compares values of different types.

    Args:
        attribute: Name of the attribute whose value is the key, the name of the type of the value if None.
        entries: Measures by key.
        fallback: Measure of the other pairs, which raise a `KeyError` if None.
        value_getter: Reads the attribute of a value, a missing attribute counts as None.

    Examples:
        >>> from cbrkit.sim.generic import Equality, Static
        >>> sim = AttributeTable(attribute="type", entries={"a": Equality()}, fallback=Static(value=0.0))
        >>> sim([({"type": "a", "v": 1}, {"type": "a", "v": 1}), ({"type": "b"}, {"type": "b"})])
        [1.0, 0.0]

        Without an attribute, the measure is chosen by the name of the type of the values:

        >>> from cbrkit.sim.numbers import Linear
        >>> by_type = AttributeTable(entries={"str": Equality(), "int": Linear(max_distance=10)})
        >>> by_type([("a", "a"), (1, 6)])
        [1.0, 0.5]
    """

    entries: Mapping[str, NestedMeasure[Any, S]]
    attribute: str | None = None
    fallback: NestedMeasure[Any, S] | None = None
    value_getter: PythonOnly[AttributeGetter] = getitem_or_getattr
    kind: Literal["attribute_table"] = field(default="attribute_table", repr=False)

    def _key(self, value: Any) -> Any:
        if self.attribute is None:
            return type(value).__name__

        return get_attribute(value, self.attribute, self.value_getter)

    def _group(self, case: Any, query: Any) -> Any:
        """The key of the measure of a pair, `_OTHER` for the fallback."""
        key = self._key(query)

        return key if key in self.entries and self._key(case) == key else _OTHER

    def _measure(self, group: Any) -> AnySimFunc[Any, S] | None:
        return self.fallback if group is _OTHER else self.entries[group]

    @override
    def __call__(self, batches: Sequence[tuple[Any, Any]]) -> SimSeq[S]:
        groups: defaultdict[Any, list[int]] = defaultdict(list)

        for idx, (case, query) in enumerate(batches):
            groups[self._group(case, query)].append(idx)

        results: dict[int, S] = {}

        for group, idxs in groups.items():
            measure = self._measure(group)

            if measure is None:
                raise KeyError(f"No measure for the pairs at {idxs}")

            sims = batchify(measure)([batches[idx] for idx in idxs])
            results.update(zip(idxs, sims, strict=True))

        return [results[idx] for idx in range(len(batches))]

    @override
    def explain(self, case: Any, query: Any, similarity: Float) -> Explanation:
        return explain(case, query, similarity, self._measure(self._group(case, query)))
