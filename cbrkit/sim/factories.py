from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd

from cbrkit.sim.helpers import aggregate_default, sim2map
from cbrkit.typing import (
    AggregateFunc,
    Casebase,
    SimilarityMap,
    SimilarityValue,
    SimMapFunc,
    SimSeqFunc,
    ValueType,
)

TabularData = Mapping[Any, Any] | pd.Series


def _key_getter(obj: TabularData) -> Iterator[str]:
    if isinstance(obj, Mapping):
        yield from obj.keys()
    elif isinstance(obj, pd.Series):
        yield from obj.index
    else:
        raise NotImplementedError()


def _value_getter(obj: TabularData, key: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj[key]
    elif isinstance(obj, pd.Series):
        return obj[key]
    else:
        return getattr(obj, key)


def equality() -> SimMapFunc[Any, Any]:
    @sim2map
    def wrapped_func(x: Any, y: Any) -> SimilarityValue:
        return x == y

    return wrapped_func


def tabular(
    attributes: Mapping[str, SimSeqFunc[Any]] | None = None,
    types: Mapping[type[Any], SimSeqFunc[Any]] | None = None,
    types_fallback: SimSeqFunc[Any] | None = None,
    aggregate: AggregateFunc = aggregate_default,
    value_getter: Callable[[Any, str], Any] = _value_getter,
    key_getter: Callable[[Any], Iterator[str]] = _key_getter,
) -> SimMapFunc[Any, TabularData]:
    attributes_map: Mapping[str, SimSeqFunc[Any]] = (
        {} if attributes is None else attributes
    )
    types_map: Mapping[type[Any], SimSeqFunc[Any]] = {} if types is None else types

    def wrapped_func(
        casebase: Casebase[Any, ValueType], query: ValueType
    ) -> SimilarityMap[Any]:
        sims_per_case: defaultdict[str, dict[str, SimilarityValue]] = defaultdict(dict)

        attribute_names = (
            set(attributes_map)
            if len(attributes_map) > 0
            and len(types_map) == 0
            and types_fallback is None
            else set(attributes_map).union(key_getter(query))
        )

        for attr in attribute_names:
            casebase_attribute_pairs = [
                (value_getter(case, attr), value_getter(query, attr))
                for case in casebase.values()
            ]
            attr_type = type(casebase_attribute_pairs[0][0])

            sim_func = (
                attributes_map[attr]
                if attr in attributes_map
                else types_map.get(attr_type, types_fallback)
            )

            assert (
                sim_func is not None
            ), f"no similarity function for {attr} with type {attr_type}"
            casebase_similarities = sim_func(casebase_attribute_pairs)

            for casename, similarity in zip(
                casebase.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregate(similarities)
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func
