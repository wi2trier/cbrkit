from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd

from cbrkit import model
from cbrkit.case_sim.helpers import aggregate_default

SupportsGetter = Mapping[Any, Any] | pd.Series


def _key_getter(obj: SupportsGetter) -> Iterator[str]:
    if isinstance(obj, Mapping):
        yield from obj.keys()
    elif isinstance(obj, pd.Series):
        yield from obj.index
    else:
        raise NotImplementedError()


def _value_getter(obj: SupportsGetter, key: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj[key]
    elif isinstance(obj, pd.Series):
        return obj[key]
    else:
        return getattr(obj, key)


def factory(
    attributes: Mapping[str, model.DataSimilarityBatchFunc[Any]] | None = None,
    types: Mapping[type[Any], model.DataSimilarityBatchFunc[Any]] | None = None,
    types_fallback: model.DataSimilarityBatchFunc[Any] | None = None,
    aggregate: model.AggregateFunc = aggregate_default,
    value_getter: Callable[[Any, str], Any] = _value_getter,
    key_getter: Callable[[Any], Iterator[str]] = _key_getter,
) -> model.CaseSimilarityBatchFunc[SupportsGetter]:
    attributes_map: Mapping[str, model.DataSimilarityBatchFunc[Any]] = (
        {} if attributes is None else attributes
    )
    types_map: Mapping[type[Any], model.DataSimilarityBatchFunc[Any]] = (
        {} if types is None else types
    )

    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

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
            casebase_similarities = sim_func(*casebase_attribute_pairs)

            for casename, similarity in zip(
                casebase.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregate(similarities)
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func
