from typing import Any

from cbrkit.case_sim.helpers import apply
from cbrkit.typing import CaseSimBatchFunc, SimilarityValue


def equality() -> CaseSimBatchFunc[Any, Any]:
    @apply
    def wrapped_func(case: Any, query: Any) -> SimilarityValue:
        return case == query

    return wrapped_func
