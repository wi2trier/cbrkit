from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from cbrkit.loaders import python as load_python
from cbrkit.sim._helpers import sim2map, unpack_sim
from cbrkit.typing import (
    AnySimFunc,
    Casebase,
    KeyType,
    RetrieveFunc,
    SimMap,
    SimType,
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
    sim_map: SimMap[KeyType, SimType],
) -> list[KeyType]:
    return sorted(sim_map, key=lambda key: unpack_sim(sim_map[key]), reverse=True)


@dataclass
class _Result(Generic[KeyType, ValueType, SimType]):
    similarities: SimMap[KeyType, SimType]
    ranking: list[KeyType]
    casebase: Casebase[KeyType, ValueType]

    @classmethod
    def build(
        cls,
        similarities: SimMap[KeyType, SimType],
        full_casebase: Casebase[KeyType, ValueType],
    ) -> "_Result[KeyType, ValueType, SimType]":
        ranking = _similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in ranking}

        return cls(similarities=similarities, ranking=ranking, casebase=casebase)


@dataclass
class Result(Generic[KeyType, ValueType, SimType]):
    _final: _Result[KeyType, ValueType, SimType]
    intermediate: list[_Result[KeyType, ValueType, SimType]]

    def __init__(
        self,
        results: list[_Result[KeyType, ValueType, SimType]],
    ) -> None:
        self.final = results[-1]
        self.intermediate = results

    @property
    def similarities(self) -> SimMap[KeyType, SimType]:
        return self.final.similarities

    @property
    def ranking(self) -> list[KeyType]:
        return self.final.ranking

    @property
    def casebase(self) -> Casebase[KeyType, ValueType]:
        return self.final.casebase


def apply(
    casebase: Casebase[KeyType, ValueType],
    query: ValueType,
    retrievers: RetrieveFunc[KeyType, ValueType, SimType]
    | Sequence[RetrieveFunc[KeyType, ValueType, SimType]],
) -> Result[KeyType, ValueType, SimType]:
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[_Result[KeyType, ValueType, SimType]] = []
    current_casebase = casebase

    for retriever_func in retrievers:
        sim_map = retriever_func(current_casebase, query)
        result = _Result.build(sim_map, current_casebase)

        results.append(result)
        current_casebase = result.casebase

    return Result(results)


def build(
    similarity_func: AnySimFunc[KeyType, ValueType, SimType],
    limit: int | None = None,
) -> RetrieveFunc[KeyType, ValueType, SimType]:
    sim_func = sim2map(similarity_func)

    def wrapped_func(
        casebase: Casebase[KeyType, ValueType],
        query: ValueType,
    ) -> SimMap[KeyType, SimType]:
        similarities = sim_func(casebase, query)
        ranking = _similarities2ranking(similarities)

        return {key: similarities[key] for key in ranking[:limit]}

    return wrapped_func


def load(
    import_names: Sequence[str] | str,
) -> list[RetrieveFunc[Any, Any, Any]]:
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
) -> dict[str, RetrieveFunc[Any, Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: dict[str, RetrieveFunc] = {}

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
