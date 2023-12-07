from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Literal, overload

from cbrkit import load, model

__all__ = ("retrieve", "retriever", "import_retrievers", "import_retrievers_map")


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


def import_retrievers(import_paths: Sequence[str] | str) -> list[model.Retriever[Any]]:
    if isinstance(import_paths, str):
        import_paths = [import_paths]

    retrievers: list[model.Retriever] = []

    for import_path in import_paths:
        obj = load.import_string(import_path)

        if isinstance(obj, Sequence):
            assert all(isinstance(func, Callable) for func in retrievers)
            retrievers.extend(obj)
        elif isinstance(obj, Callable):
            retrievers.append(obj)

    return retrievers


def import_retrievers_map(
    import_paths: Collection[str] | str
) -> dict[str, model.Retriever[Any]]:
    if isinstance(import_paths, str):
        import_paths = [import_paths]

    retrievers: dict[str, model.Retriever] = {}

    for import_path in import_paths:
        obj = load.import_string(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
