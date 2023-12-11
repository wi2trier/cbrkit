from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd

from cbrkit.sim.helpers import aggregator, sim2map, sim2seq
from cbrkit.typing import (
    AggregatorFunc,
    Casebase,
    SimMap,
    SimMapFunc,
    SimPairOrSeqFunc,
    SimType,
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
    def wrapped_func(x: Any, y: Any) -> SimType:
        return x == y

    return wrapped_func


_aggregator = aggregator()


def tabular(
    attributes: Mapping[str, SimPairOrSeqFunc[Any]] | None = None,
    types: Mapping[type[Any], SimPairOrSeqFunc[Any]] | None = None,
    types_fallback: SimPairOrSeqFunc[Any] | None = None,
    aggregator: AggregatorFunc[str] = _aggregator,
    value_getter: Callable[[Any, str], Any] = _value_getter,
    key_getter: Callable[[Any], Iterator[str]] = _key_getter,
) -> SimMapFunc[Any, TabularData]:
    attributes_map: Mapping[str, SimPairOrSeqFunc[Any]] = (
        {} if attributes is None else attributes
    )
    types_map: Mapping[type[Any], SimPairOrSeqFunc[Any]] = (
        {} if types is None else types
    )

    def wrapped_func(x_map: Casebase[Any, ValueType], y: ValueType) -> SimMap[Any]:
        sims_per_case: defaultdict[str, dict[str, SimType]] = defaultdict(dict)

        attribute_names = (
            set(attributes_map)
            if len(attributes_map) > 0
            and len(types_map) == 0
            and types_fallback is None
            else set(attributes_map).union(key_getter(y))
        )

        for attr in attribute_names:
            casebase_attribute_pairs = [
                (value_getter(case, attr), value_getter(y, attr))
                for case in x_map.values()
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

            sim_func = sim2seq(sim_func)
            casebase_similarities = sim_func(casebase_attribute_pairs)

            for casename, similarity in zip(
                x_map.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregator(similarities)
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func
