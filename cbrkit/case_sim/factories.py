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


def by_attributes(
    mapping: Mapping[str, model.DataSimilarityBatchFunc[Any]],
    aggregate: model.AggregateFunc = aggregate_default,
    value_getter: Callable[[Any, str], Any] = _value_getter,
) -> model.CaseSimilarityBatchFunc[SupportsGetter]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

        for attr, sim_func in mapping.items():
            casebase_attribute_pairs = [
                (value_getter(case, attr), value_getter(query, attr))
                for case in casebase.values()
            ]
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


def by_types(
    mapping: Mapping[type[Any], model.DataSimilarityBatchFunc[Any]],
    default_sim_func: model.DataSimilarityBatchFunc[Any] | None = None,
    aggregate: model.AggregateFunc = aggregate_default,
    value_getter: Callable[[Any, str], Any] = _value_getter,
    key_getter: Callable[[Any], Iterator[str]] = _key_getter,
) -> model.CaseSimilarityBatchFunc[SupportsGetter]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

        for attr in key_getter(query):
            casebase_attribute_pairs = [
                (value_getter(case, attr), value_getter(query, attr))
                for case in casebase.values()
            ]
            attr_type = type(casebase_attribute_pairs[0][0])
            sim_func = mapping.get(attr_type, default_sim_func)
            assert sim_func is not None, f"no similarity function for {attr_type}"
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
