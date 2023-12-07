from cbrkit import model

__all__ = ("retriever",)


def retriever(
    similarity_func: model.CaseSimilarityBatchFunc[model.CaseType],
    casebase_limit: int | None = None,
) -> model.Retriever[model.CaseType]:
    def wrapped_func(
        casebase: model.Casebase[model.CaseType],
        query: model.CaseType,
    ) -> model.RetrievalResult[model.CaseType]:
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

    return wrapped_func
