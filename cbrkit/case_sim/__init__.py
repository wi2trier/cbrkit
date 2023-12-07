from collections.abc import Collection
from typing import Any

from cbrkit import data_sim, model
from cbrkit.case_sim import factories
from cbrkit.case_sim.generic import equality
from cbrkit.case_sim.helpers import aggregate, apply

__all__ = ["get", "apply", "aggregate", "factories"]


def datatypes() -> model.CaseSimilarityBatchFunc[Any]:
    return factories.by_types(
        {
            str: data_sim.strings.levenshtein(),
            int: data_sim.numeric.exponential(),
            float: data_sim.numeric.exponential(),
            Collection: data_sim.collections.jaccard(),
        }
    )


_mapping: dict[model.SimilarityFuncName, model.CaseSimilarityBatchFunc] = {
    "equality": equality(),
    "datatypes": datatypes(),
}


def get(name: model.SimilarityFuncName) -> model.CaseSimilarityBatchFunc[Any]:
    return _mapping[name]
