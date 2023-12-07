from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import pandas as pd

from cbrkit import model
from cbrkit.case_sim.helpers import aggregate


def by_attributes(
    mapping: Mapping[str, model.DataSimilarityBatchFunc[Any]],
    pooling: model.Pooling = "mean",
    pooling_weights: Mapping[str, float] | None = None,
    default_pooling_weight: model.SimilarityValue = 1.0,
    getter: Callable[[Any, str], Any] = getattr,
) -> model.CaseSimilarityBatchFunc[Any]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

        for attr, sim_func in mapping.items():
            casebase_attribute_pairs = [
                (getter(case, attr), getter(query, attr)) for case in casebase.values()
            ]
            casebase_similarities = sim_func(*casebase_attribute_pairs)

            for casename, similarity in zip(
                casebase.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregate(
                similarities, pooling, pooling_weights, default_pooling_weight
            )
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func


def _attributes_getter(obj: Any) -> Iterator[str]:
    if isinstance(obj, Mapping):
        yield from obj.keys()
    elif isinstance(obj, pd.Series):
        yield from obj.index
    else:
        raise NotImplementedError()


def by_types(
    mapping: Mapping[type[Any], model.DataSimilarityBatchFunc[Any]],
    pooling: model.Pooling = "mean",
    pooling_weights: Mapping[str, float] | None = None,
    default_pooling_weight: model.SimilarityValue = 1.0,
    getter: Callable[[Any, str], Any] = getattr,
    attributes_getter: Callable[[Any], Iterator[str]] = _attributes_getter,
) -> model.CaseSimilarityBatchFunc[Any]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType], query: model.CaseType
    ) -> model.CaseSimilarityMap:
        sims_per_case: defaultdict[str, dict[str, model.SimilarityValue]] = defaultdict(
            dict
        )

        for attr in attributes_getter(query):
            casebase_attribute_pairs = [
                (getter(case, attr), getter(query, attr)) for case in casebase.values()
            ]
            attr_type = type(casebase_attribute_pairs[0][0])
            sim_func = mapping[attr_type]
            casebase_similarities = sim_func(*casebase_attribute_pairs)

            for casename, similarity in zip(
                casebase.keys(), casebase_similarities, strict=True
            ):
                sims_per_case[casename][attr] = similarity

        return {
            casename: aggregate(
                similarities, pooling, pooling_weights, default_pooling_weight
            )
            for casename, similarities in sims_per_case.items()
        }

    return wrapped_func
