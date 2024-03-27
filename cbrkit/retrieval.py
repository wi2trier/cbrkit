from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from cbrkit.helpers import sim2map, unpack_sim
from cbrkit.loaders import python as load_python
from cbrkit.typing import (
    AnySimFunc,
    Casebase,
    KeyType,
    SimMap,
    SimMapFunc,
    SimType,
    ValueType,
)

__all__ = [
    "build",
    "apply",
    "load",
    "load_map",
    "Result",
    "SingleResult",
]


def _similarities2ranking(
    sim_map: SimMap[KeyType, SimType],
) -> list[KeyType]:
    return sorted(sim_map, key=lambda key: unpack_sim(sim_map[key]), reverse=True)


@dataclass(slots=True)
class SingleResult(Generic[KeyType, ValueType, SimType]):
    similarities: SimMap[KeyType, SimType]
    ranking: list[KeyType]
    casebase: Casebase[KeyType, ValueType]

    @classmethod
    def build(
        cls,
        similarities: SimMap[KeyType, SimType],
        full_casebase: Casebase[KeyType, ValueType],
    ) -> "SingleResult[KeyType, ValueType, SimType]":
        ranking = _similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in ranking}

        return cls(similarities=similarities, ranking=ranking, casebase=casebase)


@dataclass(slots=True)
class Result(Generic[KeyType, ValueType, SimType]):
    final: SingleResult[KeyType, ValueType, SimType]
    intermediates: list[SingleResult[KeyType, ValueType, SimType]]

    def __init__(
        self,
        results: list[SingleResult[KeyType, ValueType, SimType]],
    ) -> None:
        self.final = results[-1]
        self.intermediates = results

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
    retrievers: SimMapFunc[KeyType, ValueType, SimType]
    | Sequence[SimMapFunc[KeyType, ValueType, SimType]],
) -> Result[KeyType, ValueType, SimType]:
    """Applies a query to a Casebase using retriever functions.

    Args:
        casebase: The casebase for the query.
        query: The query that will be applied to the casebase
        retrievers: Retriever functions that will retrieve similar cases (compared to the query) from the casebase

    Returns:
        Returns an object of type Result.

    Examples:
        >>> import cbrkit
        >>> import pandas as pd
        >>> df = pd.read_csv("./data/cars-1k.csv")
        >>> casebase = cbrkit.loaders.dataframe(df)
        >>> query = casebase[42]
        >>> retriever = cbrkit.retrieval.build(
        ...     cbrkit.sim.attribute_value(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.linear(max=100000),
        ...             "year": cbrkit.sim.numbers.linear(max=50),
        ...             "manufacturer": cbrkit.sim.strings.taxonomy.load(
        ...                 "./data/cars-taxonomy.yaml",
        ...                 measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
        ...             ),
        ...             "miles": cbrkit.sim.numbers.linear(max=1000000),
        ...         },
        ...         types_fallback=cbrkit.sim.generic.equality(),
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     ),
        ...     limit=5,
        ... )
        >>> result = cbrkit.retrieval.apply(casebase, query, retriever)
    """
    if not isinstance(retrievers, Sequence):
        retrievers = [retrievers]

    assert len(retrievers) > 0
    results: list[SingleResult[KeyType, ValueType, SimType]] = []
    current_casebase = casebase

    for retriever_func in retrievers:
        sim_map = retriever_func(current_casebase, query)
        result = SingleResult.build(sim_map, current_casebase)

        results.append(result)
        current_casebase = result.casebase

    return Result(results)


def build(
    similarity_func: AnySimFunc[KeyType, ValueType, SimType],
    limit: int | None = None,
    min_similarity: float | None = None,
    max_similarity: float | None = None,
) -> SimMapFunc[KeyType, ValueType, SimType]:
    """Based on the similarity function this function creates a retriever function.

    The given limit will be applied after filtering for min/max similarity.

    Args:
        similarity_func: Similarity function to compute the similarity between cases.
        limit: Retriever function will return the top limit cases.
        min_similarity: Return only cases with a similarity greater or equal than this.
        max_similarity: Return only cases with a similarity less or equal than this.

    Returns:
        Returns the retriever function.

    Examples:
        >>> import cbrkit
        >>> retriever = cbrkit.retrieval.build(
        ...     cbrkit.sim.attribute_value(
        ...         attributes={
        ...             "price": cbrkit.sim.numbers.linear(max=100000),
        ...             "year": cbrkit.sim.numbers.linear(max=50),
        ...             "model": cbrkit.sim.attribute_value(
        ...                 attributes={
        ...                     "make": cbrkit.sim.generic.equality(),
        ...                     "manufacturer": cbrkit.sim.strings.taxonomy.load(
        ...                         "./data/cars-taxonomy.yaml",
        ...                         measure=cbrkit.sim.strings.taxonomy.wu_palmer(),
        ...                     ),
        ...                 }
        ...             ),
        ...         },
        ...         aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ...     ),
        ...     limit=5,
        ... )
    """
    sim_func = sim2map(similarity_func)

    def wrapped_func(
        x_map: Casebase[KeyType, ValueType],
        y: ValueType,
    ) -> SimMap[KeyType, SimType]:
        similarities = sim_func(x_map, y)
        ranking = _similarities2ranking(similarities)

        if min_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) >= min_similarity
            ]
        if max_similarity is not None:
            ranking = [
                key
                for key in ranking
                if unpack_sim(similarities[key]) <= max_similarity
            ]

        return {key: similarities[key] for key in ranking[:limit]}

    return wrapped_func


def load(
    import_names: Sequence[str] | str,
) -> list[SimMapFunc[Any, Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: list[SimMapFunc] = []

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
) -> dict[str, SimMapFunc[Any, Any, Any]]:
    if isinstance(import_names, str):
        import_names = [import_names]

    retrievers: dict[str, SimMapFunc] = {}

    for import_path in import_names:
        obj = load_python(import_path)

        if isinstance(obj, Mapping):
            assert all(isinstance(func, Callable) for func in obj.values())
            retrievers.update(obj)

    return retrievers
