from cbrkit import model


def equality(case: model.CaseType, query: model.CaseType) -> model.SimilarityValue:
    return case == query


_mapping: dict[model.SimilarityType, model.SimilarityFunc] = {
    "equality": equality,
}


def get(name: model.SimilarityType) -> model.SimilarityFunc[model.CaseType]:
    return _mapping[name]
