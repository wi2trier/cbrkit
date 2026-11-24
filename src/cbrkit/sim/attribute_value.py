import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, override

import numpy as np
import numpy.typing as npt

from ..helpers import batchify, frozendict, get_attribute, getitem_or_getattr
from ..model.explanation import Explanation
from ..model.similarity import AttributeValueSim, SerializedFloat
from ..schema import PythonOnly, serializable
from ..typing import AttributeGetter, BatchSimFunc, NumpyArray, SimSeq
from ._explain import Explainable, explain_attributes
from ._nested import NestedMeasure
from .aggregator import Aggregator, default_aggregator

__all__ = ["AttributeValue"]

logger = logging.getLogger(__name__)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AttributeValue[V, S: SerializedFloat](
    BatchSimFunc[V, AttributeValueSim[S]], Explainable[V]
):
    """Similarity function that computes the attribute value similarity between two cases.

    Missing attributes (lookup errors of the `value_getter`) are treated like `None` values.
    Every attribute is scored as one column of pairs, so measures implementing `cbrkit.typing.ColumnSimFunc`
    such as the numeric, date, string and equality measures run vectorised,
    and an `Aggregator` aggregates all pairs at once (see `Aggregator.columns`).

    Args:
        attributes: A mapping of attribute names to the similarity functions to be used for those attributes.
        aggregator: Aggregates the local similarities of the attributes into the global one.
        value_getter: A function that retrieves the value of an attribute from a case.
        default: The default similarity score of a pair whose attribute is missing or whose local similarity fails.
            If None, such pairs raise an error instead.
        skip_missing_query: If True, attributes that are missing in the query are left out of the aggregation,
            so the aggregator renormalizes the weights of the remaining attributes (partial queries).

    Examples:
        >>> equality = lambda case, query: 1.0 if case == query else 0.0
        >>> sim = AttributeValue(attributes={
        ...     "name": equality,
        ...     "age": equality,
        ... })
        >>> scores = sim([
        ...     ({"name": "John", "age": 25}, {"name": "John", "age": 30}),
        ...     ({"name": "Jane", "age": 30}, {"name": "John", "age": 30}),
        ... ])
        >>> scores[0]
        AttributeValueSim(value=0.5, similarities=frozendict({'name': 1.0, 'age': 0.0}))
        >>> scores[1]
        AttributeValueSim(value=0.5, similarities=frozendict({'name': 0.0, 'age': 1.0}))
        >>> partial = AttributeValue(attributes={"name": equality, "age": equality}, skip_missing_query=True)
        >>> partial([({"name": "John", "age": 25}, {"name": "John"})])[0]
        AttributeValueSim(value=1.0, similarities=frozendict({'name': 1.0}))
    """

    attributes: Mapping[str, NestedMeasure[Any, S]]
    aggregator: Aggregator = default_aggregator
    value_getter: PythonOnly[AttributeGetter] = getitem_or_getattr
    default: S | None = None
    skip_missing_query: bool = False
    kind: Literal["attribute_value"] = field(default="attribute_value", repr=False)

    def _values(self, objs: Sequence[Any], attr_name: str) -> list[Any]:
        try:
            # item access without a check per value for the default getter
            if self.value_getter is getitem_or_getattr:
                return [obj[attr_name] for obj in objs]

            return [self.value_getter(obj, attr_name) for obj in objs]
        except LookupError, AttributeError, TypeError:
            return [get_attribute(obj, attr_name, self.value_getter) for obj in objs]

    def _fallback(self, attr_name: str, idx: int) -> S:
        if self.default is None:
            raise KeyError(f"Attribute '{attr_name}' is missing at index {idx}")

        return self.default

    def _column(
        self,
        attr_name: str,
        measure: BatchSimFunc[Any, S],
        cases: Sequence[Any],
        queries: Sequence[Any],
    ) -> tuple[list[S | None], npt.NDArray[np.bool_]]:
        """The local similarity of every pair and whether it is part of the aggregation."""
        column: list[S | None] = [None] * len(cases)
        present = np.ones(len(cases), dtype=np.bool_)
        missing = {
            idx
            for idx, (x_val, y_val) in enumerate(zip(cases, queries, strict=True))
            if x_val is None or y_val is None
        }

        for idx in sorted(missing):
            if queries[idx] is None and self.skip_missing_query:
                present[idx] = False
            else:
                column[idx] = self._fallback(attr_name, idx)

        if not missing:
            return list(self._similarities(measure, list(zip(cases, queries)))), present

        indices = [idx for idx in range(len(cases)) if idx not in missing]

        if indices:
            pairs = [(cases[idx], queries[idx]) for idx in indices]

            for idx, sim in zip(
                indices, self._similarities(measure, pairs), strict=True
            ):
                column[idx] = sim

        return column, present

    def _similarities(
        self, measure: BatchSimFunc[Any, S], pairs: Sequence[tuple[Any, Any]]
    ) -> SimSeq[S]:
        try:
            return measure(pairs)
        except Exception:
            if self.default is None:
                raise

            # retry pairwise so that one failing pair does not affect the others
            return [_pair_sim(measure, pair, self.default) for pair in pairs]

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[AttributeValueSim[S]]:
        if not batches:
            return []

        cases = [x for x, _ in batches]
        queries = [y for _, y in batches]
        names = list(self.attributes)
        columns: list[list[S | None]] = []
        masks: list[npt.NDArray[np.bool_]] = []

        for attr_name, attr_func in self.attributes.items():
            logger.debug("Processing attribute %s", attr_name)
            column, mask = self._column(
                attr_name,
                batchify(attr_func),
                self._values(cases, attr_name),
                self._values(queries, attr_name),
            )
            columns.append(column)
            masks.append(mask)

        present = np.array(masks, dtype=np.bool_).reshape(len(names), len(batches)).T
        # Without attributes, every pair still gets its empty local similarities.
        rows: Iterable[tuple[S | None, ...]] = (
            zip(*columns, strict=True) if columns else [()] * len(batches)
        )
        local_sims: list[frozendict[str, S]] = (
            [frozendict(zip(names, row)) for row in rows]
            if present.all()
            else [
                frozendict(
                    (name, sim)
                    for name, sim, keep in zip(names, row, mask, strict=True)
                    if keep
                )
                for row, mask in zip(rows, present.tolist(), strict=True)
            ]
        )
        global_sims = self.aggregator.columns(
            names, _matrix(columns, present), present
        ).tolist()

        return [
            AttributeValueSim(value=value, similarities=sims)
            for value, sims in zip(global_sims, local_sims, strict=True)
        ]

    @override
    def explain(self, case: V, query: V, similarity: SerializedFloat) -> Explanation:
        return explain_attributes(
            similarity,
            self.aggregator,
            self.attributes,
            lambda name: (
                get_attribute(case, name, self.value_getter),
                get_attribute(query, name, self.value_getter),
            ),
        )


def _matrix[S: SerializedFloat](
    columns: Sequence[Sequence[S | None]], present: npt.NDArray[np.bool_]
) -> NumpyArray:
    """The similarities as floats with one row per pair, 0.0 where they are left out."""
    rows = [
        np.asarray(column, dtype=np.float64)
        if mask.all()
        else np.array(
            [0.0 if sim is None else float(sim) for sim in column], dtype=np.float64
        )
        for column, mask in zip(columns, present.T, strict=True)
    ]

    return np.array(rows, dtype=np.float64).reshape(present.shape[::-1]).T


def _pair_sim[S: SerializedFloat](
    measure: BatchSimFunc[Any, S], pair: tuple[Any, Any], default: S
) -> S:
    try:
        return measure([pair])[0]
    except Exception:
        logger.warning("Using the default similarity for a failing pair", exc_info=True)
        return default
