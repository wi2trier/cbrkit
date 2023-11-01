from cbrkit import model, similarity


def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    similarity_func: model.SimilarityType
    | model.SimilarityFunc[model.CaseType] = similarity.equality,
    type: model.RetrievalType = "linear",
    casebase_limit: int | None = None,
) -> model.RetrievalResult[model.CaseType]:
    if isinstance(similarity_func, str):
        similarity_func = similarity.get(similarity_func)

    result: model.RetrievalResult[model.CaseType] | None = None

    if type == "linear":
        result = _retrieve_linear(casebase, query, similarity_func)
    else:
        raise NotImplementedError()

    assert result is not None

    if casebase_limit is not None:
        result.casebase = {
            key: result.casebase[key] for key in result.ranking[:casebase_limit]
        }

    return result


def _retrieve_linear(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    similarity_func: model.SimilarityFunc[model.CaseType],
) -> model.RetrievalResult[model.CaseType]:
    similarities = {key: similarity_func(case, query) for key, case in casebase.items()}
    ranked_tuples = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    ranking = [key for key, _ in ranked_tuples]

    return model.RetrievalResult(
        similarities=similarities,
        ranking=ranking,
        casebase=casebase,
    )
