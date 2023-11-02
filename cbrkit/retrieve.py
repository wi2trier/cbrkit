from cbrkit import model, similarity


def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    similarity_func: model.SimilarityType
    | model.SimilarityFunc[model.CaseType] = similarity.equality,
    parallel: bool = False,
    casebase_limit: int | None = None,
) -> model.RetrievalResult[model.CaseType]:
    if isinstance(similarity_func, str):
        similarity_func = similarity.get(similarity_func)

    similarities: dict[model.CaseName, model.SimilarityValue]

    if parallel:
        raise NotImplementedError()
    else:
        similarities = {
            key: similarity_func(case, query) for key, case in casebase.items()
        }

    ranked_tuples = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    ranking = [key for key, _ in ranked_tuples]
    filtered_casebase = (
        casebase
        if casebase_limit is None
        else {key: casebase[key] for key in ranking[:casebase_limit]}
    )

    return model.RetrievalResult(
        similarities=similarities, ranking=ranking, casebase=filtered_casebase
    )
