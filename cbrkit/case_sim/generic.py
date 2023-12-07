from typing import Any

from cbrkit import model
from cbrkit.case_sim.helpers import apply


def equality() -> model.CaseSimilarityBatchFunc[Any]:
    @apply
    def wrapped_func(case: Any, query: Any) -> model.SimilarityValue:
        return case == query

    return wrapped_func
