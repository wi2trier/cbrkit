from collections.abc import Callable, Sequence
from typing import Any, Literal, overload

from cbrkit import load, model

__all__ = ("retrieve", "retriever", "import_retrievers")


@overload
def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    retrievers: model.Retriever[model.CaseType]
    | Sequence[model.Retriever[model.CaseType]],
    all_results: Literal[False] = False,
) -> model.RetrievalResult[model.CaseType]:
    ...


@overload
def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    retrievers: model.Retriever[model.CaseType]
    | Sequence[model.Retriever[model.CaseType]],
    all_results: Literal[True] = True,
) -> list[model.RetrievalResult[model.CaseType]]:
    ...


def retrieve(
    casebase: model.Casebase[model.CaseType],
    query: model.CaseType,
    retrievers: model.Retriever[model.CaseType]
    | Sequence[model.Retriever[model.CaseType]],
    all_results: bool = False,
) -> (
    model.RetrievalResult[model.CaseType] | list[model.RetrievalResult[model.CaseType]]
):
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[model.RetrievalResult[model.CaseType]] = []
    current_casebase = casebase

    for retriever_func in retrievers:
        result = retriever_func(current_casebase, query)
        current_casebase = result.casebase
        results.append(result)

    if all_results:
        return results
    else:
        return results[-1]


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


def import_retrievers(retriever: str) -> Sequence[model.Retriever[Any]]:
    retriever_funcs: model.Retriever | Sequence[model.Retriever] = load.import_string(
        retriever
    )

    if not isinstance(retriever_funcs, Sequence):
        retriever_funcs = [retriever_funcs]

    assert all(isinstance(func, Callable) for func in retriever_funcs)

    return retriever_funcs
