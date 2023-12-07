from cbrkit import model
from cbrkit.case_sim.helpers import apply


def equality() -> model.CaseSimilarityBatchFunc[model.CaseType]:
    @apply
    def wrapped_func(
        case: model.CaseType, query: model.CaseType
    ) -> model.SimilarityValue:
        return case == query

    return wrapped_func
