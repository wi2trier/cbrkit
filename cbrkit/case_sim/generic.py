from typing import Any

from cbrkit.case_sim.helpers import batchify
from cbrkit.typing import CaseSimBatchFunc, SimilarityValue


def equality() -> CaseSimBatchFunc[Any, Any]:
    @batchify
    def wrapped_func(case: Any, query: Any) -> SimilarityValue:
        return case == query

    return wrapped_func
