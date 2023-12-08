from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from cbrkit import load
from cbrkit.typing import (
    Casebase,
    CaseName,
    CaseSimBatchFunc,
    CaseType,
    RetrievalResultProtocol,
    RetrieveFunc,
    SimilarityMap,
)

__all__ = ("retrieve", "retriever", "import_retrievers", "import_retrievers_map")


@dataclass
class RetrievalResult(RetrievalResultProtocol[CaseName, CaseType]):
    similarities: SimilarityMap[CaseName]
    ranking: list[CaseName]
    casebase: Casebase[CaseName, CaseType]


@overload
def retrieve(
    casebase: Casebase[CaseName, CaseType],
    query: CaseType,
    retrievers: RetrieveFunc[CaseName, CaseType]
    | Sequence[RetrieveFunc[CaseName, CaseType]],
    all_results: Literal[False] = False,
) -> RetrievalResultProtocol[CaseName, CaseType]:
    ...


@overload
def retrieve(
    casebase: Casebase[CaseName, CaseType],
    query: CaseType,
    retrievers: RetrieveFunc[CaseName, CaseType]
    | Sequence[RetrieveFunc[CaseName, CaseType]],
    all_results: Literal[True] = True,
) -> list[RetrievalResultProtocol[CaseName, CaseType]]:
    ...


def retrieve(
    casebase: Casebase[CaseName, CaseType],
    query: CaseType,
    retrievers: RetrieveFunc[CaseName, CaseType]
    | Sequence[RetrieveFunc[CaseName, CaseType]],
    all_results: bool = False,
) -> (
    RetrievalResultProtocol[CaseName, CaseType]
    | list[RetrievalResultProtocol[CaseName, CaseType]]
):
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[RetrievalResultProtocol[CaseName, CaseType]] = []
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
    similarity_func: CaseSimBatchFunc[CaseName, CaseType],
    casebase_limit: int | None = None,
) -> RetrieveFunc[CaseName, CaseType]:
    def wrapped_func(
        casebase: Casebase[CaseName, CaseType],
        query: CaseType,
    ) -> RetrievalResultProtocol[CaseName, CaseType]:
        similarities = similarity_func(casebase, query)

        ranked_tuples = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        ranking = [key for key, _ in ranked_tuples]
        filtered_casebase = (
            casebase
            if casebase_limit is None
            else {key: casebase[key] for key in ranking[:casebase_limit]}
        )

        return RetrievalResult(
            similarities=similarities, ranking=ranking, casebase=filtered_casebase
        )

    return wrapped_func


def import_retrievers(
    import_names: Sequence[str] | str
) -> list[RetrieveFunc[Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: list[RetrieveFunc] = []

    for import_path in import_names:
        obj = load.python(import_path)

        if isinstance(obj, Sequence):
            assert all(isinstance(func, Callable) for func in retrievers)
            retrievers.extend(obj)
        elif isinstance(obj, Callable):
            retrievers.append(obj)

    return retrievers


def import_retrievers_map(
    import_names: Collection[str] | str
) -> dict[str, RetrieveFunc[Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: dict[str, RetrieveFunc] = {}

    for import_path in import_names:
        obj = load.python(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
