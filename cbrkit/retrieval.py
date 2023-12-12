from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, overload

from cbrkit.loaders import python as load_python
from cbrkit.sim._helpers import sim2map
from cbrkit.typing import (
    AnySimFunc,
    Casebase,
    KeyType,
    RetrieveFunc,
    SimMap,
    ValueType,
)

__all__ = [
    "build",
    "apply",
    "load",
    "load_map",
    "Result",
]


def _similarities2ranking(
    sim_map: SimMap[KeyType],
) -> list[KeyType]:
    return sorted(sim_map, key=lambda key: sim_map[key], reverse=True)


@dataclass
class Result(Generic[KeyType, ValueType]):
    similarities: SimMap[KeyType]
    ranking: list[KeyType]
    casebase: Casebase[KeyType, ValueType]

    @classmethod
    def build(
        cls, similarities: SimMap[KeyType], full_casebase: Casebase[KeyType, ValueType]
    ) -> "Result[KeyType, ValueType]":
        ranking = _similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in ranking}

        return cls(similarities=similarities, ranking=ranking, casebase=casebase)


@overload
def apply(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: Literal[False] = False,
) -> Result[KeyType, ValueType]:
    ...


@overload
def apply(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: Literal[True] = True,
) -> list[Result[KeyType, ValueType]]:
    ...


def apply(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType]
    | Sequence[RetrieveFunc[KeyType, ValueType]],
    all_results: bool = False,
) -> Result[KeyType, ValueType] | list[Result[KeyType, ValueType]]:
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[Result[KeyType, ValueType]] = []
    current_casebase = casebase

    for retriever_func in retrievers:
        sim_map = retriever_func(current_casebase, query)
        result = Result.build(sim_map, current_casebase)

        results.append(result)
        current_casebase = result.casebase

    if all_results:
        return results
    else:
        return results[-1]


def build(
    similarity_func: AnySimFunc[KeyType, ValueType],
    limit: int | None = None,
) -> RetrieveFunc[KeyType, ValueType]:
    sim_func = sim2map(similarity_func)

    def wrapped_func(
        casebase: Casebase[KeyType, ValueType],
        query: ValueType,
    ) -> SimMap[KeyType]:
        similarities = sim_func(casebase, query)
        ranking = _similarities2ranking(similarities)

        return {key: similarities[key] for key in ranking[:limit]}

    return wrapped_func


def load(
    import_names: Sequence[str] | str,
) -> list[RetrieveFunc[Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: list[RetrieveFunc] = []

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Sequence):
            assert all(isinstance(func, Callable) for func in retrievers)
            retrievers.extend(obj)
        elif isinstance(obj, Callable):
            retrievers.append(obj)

    return retrievers


def load_map(
    import_names: Collection[str] | str,
) -> dict[str, RetrieveFunc[Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: dict[str, RetrieveFunc] = {}

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
