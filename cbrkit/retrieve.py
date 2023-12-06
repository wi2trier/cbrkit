from collections.abc import Callable

from cbrkit import model, similarity

__all__ = ("retrieve",)


def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    similarity_func: model.SimilarityFuncName
    | model.CaseSimilarityBatchFunc[model.CaseType],
    casebase_limit: int | None = None,
) -> model.RetrievalResult[model.CaseType]:
    if not isinstance(similarity_func, Callable):
        similarity_func = similarity.get(similarity_func)

    similarities = similarity_func(casebase, query)

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
