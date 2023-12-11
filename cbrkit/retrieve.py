from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, overload

from cbrkit import load
from cbrkit.sim._helpers import sim2map
from cbrkit.typing import (
    AnySimFunc,
    Casebase,
    KeyType,
    RetrievalResultProtocol,
    RetrieveFunc,
    SimMap,
    ValueType,
)

__all__ = ("retrieve", "retriever", "import_retrievers", "import_retrievers_map")


@dataclass
class RetrievalResult(RetrievalResultProtocol[KeyType, ValueType]):
    similarities: SimMap[KeyType]
    ranking: list[KeyType]
    casebase: Casebase[KeyType, ValueType]


@overload
def retrieve(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: Literal[False] = False,
) -> RetrievalResultProtocol[KeyType, ValueType]:
    ...


@overload
def retrieve(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: Literal[True] = True,
) -> list[RetrievalResultProtocol[KeyType, ValueType]]:
    ...


def retrieve(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: bool = False,
) -> (
    RetrievalResultProtocol[KeyType, ValueType]
    | list[RetrievalResultProtocol[KeyType, ValueType]]
):
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[RetrievalResultProtocol[KeyType, ValueType]] = []
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
    similarity_func: AnySimFunc[KeyType, ValueType],
    casebase_limit: int | None = None,
) -> RetrieveFunc[KeyType, ValueType]:
    sim_func = sim2map(similarity_func)

    def wrapped_func(
        casebase: Casebase[KeyType, ValueType],
        query: ValueType,
    ) -> RetrievalResultProtocol[KeyType, ValueType]:
        similarities = sim_func(casebase, query)

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
    import_names: Sequence[str] | str,
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
    import_names: Collection[str] | str,
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
