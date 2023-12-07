from cbrkit import model
from cbrkit.case_sim.generic import equality

_mapping: dict[model.SimilarityFuncName, model.CaseSimilarityBatchFunc] = {
    "equality": equality(),
}


def get(
    name: model.SimilarityFuncName
) -> model.CaseSimilarityBatchFunc[model.CaseType]:
    return _mapping[name]
