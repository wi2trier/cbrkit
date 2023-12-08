from typing import Any

from cbrkit.case_sim.helpers import apply
from cbrkit.typing import CasebaseSimFunc, SimilarityValue


def equality() -> CasebaseSimFunc[Any, Any]:
    @apply
    def wrapped_func(case: Any, query: Any) -> SimilarityValue:
        return case == query

    return wrapped_func
