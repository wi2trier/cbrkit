"""Protocols and type aliases of the functions and data of cbrkit, every one replaceable by a custom implementation.

They are the contracts that cbrkit implements and depend on the standard library and numpy only,
how configurations and results serialize is up to `cbrkit.schema` and `cbrkit.model`.

Cases and queries are always of the same type `V`, any value is its own problem and solution,
and `cbrkit.model.Case` splits a case into a problem and a solution, see `cbrkit.model`.

- Similarities: `Float` is the bound of every similarity, a float or a `Sim` with details.
- Pairs: `PairFunc` and `BatchPairFunc` are the shapes of the measures and adaptations, marked as batch functions by `Batched`.
- Measures: `SimFunc` and `BatchSimFunc` compare a case with a query, `AnySimFunc` is either of them,
  `ColumnSimFunc` and `MatrixSimFunc` score many pairs at once with numpy,
  and `AttributeGetter` and `PoolingFunc` are their building blocks.
- Adaptations: `AdaptationFunc` and `BatchAdaptationFunc` adapt a case to a query, `AnyAdaptationFunc` is either of them.
- The cycle: `CbrFunc` and `AsyncCbrFunc` are its functions over `Batches` with `Results`,
  the retrievers, reusers, revisers, and retainers.
- Effects: `EmbeddingFunc`, `GenerationFunc`, and `RerankFunc` are the models of the host.
- Storage: `Index`, `FilterableIndex`, and `Storage` hold the cases.
- Data: `Casebase`, `SimMap`, `JsonEntry`, and the other aliases.
"""

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, SupportsFloat, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "AdaptationFunc",
    "AnyAdaptationFunc",
    "AnyCbrFunc",
    "AnySimFunc",
    "AsyncCbrFunc",
    "AttributeGetter",
    "BatchAdaptationFunc",
    "BatchPairFunc",
    "BatchSimFunc",
    "Batched",
    "Batches",
    "Casebase",
    "CbrFunc",
    "ColumnPoolingFunc",
    "ColumnSimFunc",
    "ConversionFunc",
    "EmbeddingFunc",
    "EvalMetricFunc",
    "FilePath",
    "FilterableIndex",
    "Float",
    "GenerationFunc",
    "GroupCasebase",
    "HasValue",
    "Index",
    "JsonDict",
    "JsonEntry",
    "MatrixSimFunc",
    "NumpyArray",
    "PairFunc",
    "PoolingFunc",
    "QueryCaseMatrix",
    "RerankFunc",
    "Results",
    "Sim",
    "SimFunc",
    "SimMap",
    "SimSeq",
    "SparseVector",
    "Storage",
    "WeightedPoolingFunc",
]

type JsonEntry = (
    Mapping[str, JsonEntry] | Sequence[JsonEntry] | str | int | float | bool | None
)
"""A JSON value: an object, an array, a string, a number, a boolean, or null."""
type JsonDict = dict[str, JsonEntry]
"""A JSON object by its keys."""
type NumpyArray = npt.NDArray[np.float64]
"""A numpy array of floats, such as an embedding."""
type SparseVector = dict[int, float]
"""A sparse vector of its nonzero values by dimension, such as a BM25 or SPLADE embedding."""


class HasValue[T](Protocol):
    """An object holding a value, such as a graph node or a provider response."""

    @property
    def value(self) -> T:
        """The held value."""
        ...


class Index[K, V](Protocol):
    """Cases that are read and replaced, asynchronously since a storage does I/O.

    It holds the cases of a database, the index of a retriever, or the cases that a run reads once.
    Every storage of `cbrkit.indexable` implements it, those that add cases with keys they allocate
    implement `Storage` as well.
    """

    async def get_index(self) -> Casebase[K, V]:
        """Return the current cases."""
        ...

    async def has_index(self) -> bool:
        """Return whether the cases have been created."""
        ...

    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Replace all cases with the given ones."""
        ...


class FilterableIndex[K, V, W](Index[K, V], Protocol):
    """`Index` extended with WHERE-style mutators.

    `W` is the backend's native predicate type (`sa.ColumnElement[bool]`
    for the SQLAlchemy storages, a SQL string for LanceDB), passed through
    to the backend unchanged.
    """

    async def keys_where(self, where: W, /) -> Collection[K]:
        """Return keys matching the given filter predicate."""
        ...

    async def get_where(self, where: W, /) -> Casebase[K, V]:
        """Return the typed cases matching the given filter predicate, such as the rows of retrieved keys."""
        ...

    async def delete_where(self, where: W, /) -> Collection[K]:
        """Delete rows matching the filter, returning the affected keys."""
        ...

    async def replace_where(self, where: W, data: Casebase[K, V], /) -> Collection[K]:
        """Atomically replace the subset matched by `where` with `data`."""
        ...


@dataclass(slots=True, frozen=True, kw_only=True)
class Sim:
    """Similarity value with details, usable wherever a float is expected.

    `float(sim)` returns the value and comparisons with numbers use it,
    while equality also compares the details.
    Subclasses such as `cbrkit.model.AttributeValueSim` add details,
    which results keep through JSON by their kind, see `cbrkit.model.SerializedFloat`.

    Args:
        value: The similarity.

    Examples:
        >>> sim = Sim(value=0.5)
        >>> float(sim), sim < 0.75, sorted([1.0, sim])
        (0.5, True, [Sim(value=0.5), 1.0])
    """

    value: float

    def __float__(self) -> float:
        return self.value

    def __lt__(self, other: SupportsFloat) -> bool:
        return self.value < float(other)

    def __le__(self, other: SupportsFloat) -> bool:
        return self.value <= float(other)

    def __gt__(self, other: SupportsFloat) -> bool:
        return self.value > float(other)

    def __ge__(self, other: SupportsFloat) -> bool:
        return self.value >= float(other)


type Float = float | Sim
"""A plain or a detailed similarity, the bound of every similarity type.

`float(sim)` gives the number of both.
"""


type FilePath = str | Path
"""A path of a file or folder, as a string or a `pathlib.Path`."""
type Casebase[K, V] = Mapping[K, V]
"""The cases by their keys, split into problem and solution if they are `cbrkit.model.Case`s."""
type GroupCasebase[K, V] = Casebase[tuple[K, ...], tuple[V, ...]]
"""A casebase whose cases are groups of source cases, keyed by their source keys."""
type SimMap[K, S: Float = float] = Mapping[K, S]
"""The similarities of the cases by their keys."""
type SimSeq[S: Float = float] = Sequence[S]
"""The similarities of a sequence of pairs, in their order."""
type QueryCaseMatrix[Q, C, V] = Mapping[Q, Mapping[C, V]]
"""Values by query key and case key, such as the similarities of every query to every case."""
type Batches[K, V] = Sequence[tuple[Casebase[K, V], V]]
"""The (casebase, query) pairs that a function of the CBR cycle takes, see `CbrFunc`."""
type Results[K, V, S: Float = float] = Sequence[tuple[Casebase[K, V], SimMap[K, S]]]
"""The cases that a function of the CBR cycle keeps for every batch with their scores, see `CbrFunc`."""


class Batched(Protocol):
    """Marker base of the batch protocols such as `BatchSimFunc`.

    Classes deriving from a batch protocol are batch functions,
    and plain functions become ones with the `cbrkit.helpers.batched` decorator.
    `cbrkit.helpers.batchify` and `cbrkit.helpers.unbatchify` rely on this marking
    instead of guessing by the signature.
    """


class PairFunc[V, R](Protocol):
    """A function of a case and a query, such as a similarity or an adaptation function."""

    def __call__(self, case: V, query: V, /) -> R:
        """Computes the result for the case and the query."""
        ...


class BatchPairFunc[V, R](Batched, Protocol):
    """A function of many (case, query) pairs at once, see `cbrkit.helpers.batchify`."""

    def __call__(self, batches: Sequence[tuple[V, V]], /) -> Sequence[R]:
        """Computes the result for every (case, query) pair, in their order."""
        ...


class ConversionFunc[U, V](Protocol):
    """Converts a single value from type U to type V."""

    def __call__(
        self,
        value: U,
        /,
    ) -> V:
        """Converts the value."""
        ...


class EmbeddingFunc[V, E](Protocol):
    """Embeds many values at once, such as the texts of a provider request.

    It is asynchronous like every function with effects, since a provider is remote or runs a model.
    Measures that embed, such as `cbrkit.sim.embed.Embedding`, wait for it with `cbrkit.helpers.wait`.
    """

    async def __call__(
        self,
        values: Sequence[V],
        /,
    ) -> Sequence[E]:
        """Embeds the values, in their order."""
        ...


class SimFunc[V, S: Float = float](PairFunc[V, S], Protocol):
    """Computes similarity between two values of type V."""

    def __call__(
        self,
        case: V,
        query: V,
        /,
    ) -> S:
        """Computes the similarity of the case to the query."""
        ...


class BatchSimFunc[V, S: Float = float](BatchPairFunc[V, S], Protocol):
    """Computes similarities for multiple value pairs in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> SimSeq[S]:
        """Computes the similarity of every (case, query) pair, in their order."""
        ...


@runtime_checkable
class ColumnSimFunc[V](SimFunc[V, float], Protocol):
    """A similarity function that also scores whole columns of value pairs at once.

    The built-in numeric, date, string and generic measures implement it with numpy or rapidfuzz,
    and `cbrkit.helpers.batchify` calls `columns` instead of the pair function for every pair.
    The pair function stays the reference, `columns` returns the same similarities within float tolerance.
    """

    def columns(
        self,
        cases: Sequence[V],
        queries: Sequence[V],
        /,
    ) -> NumpyArray:
        """Return the similarity of every case to the query at the same position."""
        ...


@runtime_checkable
class MatrixSimFunc(SimFunc[NumpyArray, float], Protocol):
    """A similarity function of vectors that also scores every case against every query at once.

    The dense metrics of `cbrkit.sim.embed` implement it,
    and `cbrkit.retrieve.Embedding` scores a whole case base with it.
    """

    def matrix(
        self,
        cases: NumpyArray,
        queries: NumpyArray,
        /,
    ) -> NumpyArray:
        """Return the similarities of the rows of `cases` (n, d) to the rows of `queries` (m, d) as (n, m)."""
        ...


type AttributeGetter = Callable[[Any, str], Any]
"""Reads an attribute of a value by its name, raising a `LookupError`, `AttributeError`, or `TypeError` if it is missing."""


type AnySimFunc[V, S: Float = float] = SimFunc[V, S] | BatchSimFunc[V, S]
"""A similarity or batch similarity function."""


class CbrFunc[K, V, S: Float = float](Protocol):
    """A function of the CBR cycle: a retriever, reuser, reviser, or retainer.

    It takes a sequence of (casebase, query) batches
    and returns the cases it keeps for every batch with their scores.
    The returned cases may differ from the given ones,
    for instance adapted cases in reuse or the cases as stored in retain.
    Queries are of the type of the cases, `cbrkit.model.Case`s without a solution for split cases.
    A function that is generic over the keys declares the key type on its `__call__`,
    such as `cbrkit.retrieve.Retriever`, so it is typed without a casebase.
    """

    def __call__(self, batches: Batches[K, V], /) -> Results[K, V, S]:
        """Returns the cases that the function keeps for every batch with their scores, in the order of the batches."""
        ...


class AsyncCbrFunc[K, V, S: Float = float](Protocol):
    """A function of the CBR cycle with effects, for instance a retriever of a database, a reranker, or a retainer.

    Pure functions such as `cbrkit.retrieve.Retriever` implement `CbrFunc`,
    which `cbrkit.cycle.run_async` runs in a worker thread.
    """

    async def __call__(self, batches: Batches[K, V], /) -> Results[K, V, S]:
        """Returns the cases that the function keeps for every batch with their scores, in the order of the batches."""
        ...


type AnyCbrFunc[K, V, S: Float = float] = CbrFunc[K, V, S] | AsyncCbrFunc[K, V, S]
"""A synchronous or asynchronous function of the CBR cycle."""


class AdaptationFunc[V](PairFunc[V, V], Protocol):
    """Adapts a case to match a query."""

    def __call__(
        self,
        case: V,
        query: V,
        /,
    ) -> V:
        """Adapts the case to the query."""
        ...


class BatchAdaptationFunc[V](BatchPairFunc[V, V], Protocol):
    """Adapts multiple case-query pairs in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> Sequence[V]:
        """Adapts the case of every (case, query) pair to its query, in their order."""
        ...


type AnyAdaptationFunc[V] = AdaptationFunc[V] | BatchAdaptationFunc[V]
"""An adaptation or batch adaptation function."""


class RerankFunc(Protocol):
    """Scores documents for a query with a rerank model, such as a cross-encoder of the host.

    It returns the scores by the positions of the documents and may leave out documents,
    such as those beyond the `limit` best ones that an endpoint returns.
    """

    async def __call__(
        self,
        query: str,
        documents: Sequence[str],
        /,
        limit: int | None = None,
    ) -> Mapping[int, float]:
        """Scores the documents for the query by their positions, at least the `limit` best ones, all if None."""
        ...


class GenerationFunc(Protocol):
    """Generates a structured response for every prompt, such as a language model of the host.

    It is asynchronous like every function with effects,
    and measures and adaptations that generate, such as `cbrkit.sim.llm.Critique`, wait for it with `cbrkit.helpers.wait`.
    """

    async def __call__[T](
        self,
        prompts: Sequence[str],
        response_type: type[T],
        /,
    ) -> Sequence[T]:
        """Generates a response of the type for every prompt, in their order."""
        ...


class PoolingFunc[T](Protocol):
    """Pools multiple values into a single representative value."""

    def __call__(
        self,
        values: Sequence[T],
        /,
    ) -> T:
        """Pools the values into one."""
        ...


@runtime_checkable
class WeightedPoolingFunc(PoolingFunc[float], Protocol):
    """A pooling function with a weighted variant, which `cbrkit.sim.Aggregator` calls with weights.

    Pooling functions without it receive the values multiplied by their weights.
    """

    def weighted(self, values: Sequence[float], weights: Sequence[float], /) -> float:
        """Pool the values with positive weights normalized to a mean of 1.

        Uniform weights give the same result as the unweighted pooling function.
        """
        ...


@runtime_checkable
class ColumnPoolingFunc(WeightedPoolingFunc, Protocol):
    """A weighted pooling function that also pools every row of a matrix at once with numpy."""

    def columns(
        self, values: NumpyArray, weights: NumpyArray, kept: npt.NDArray[np.bool_], /
    ) -> NumpyArray:
        """Pool the kept values of every row with their weights, normalized to a mean of 1 per row.

        Values and weights that are not kept are 0, a row without kept values may give any result.
        """
        ...


class EvalMetricFunc(Protocol):
    """Evaluates the retrieved cases of one query against its relevance judgments, such as `cbrkit.eval.Metric`.

    Cases are given by their text,
    and a metric is hashable since it keys the results of `cbrkit.eval.compute`, which aggregates it over the queries.
    """

    def __call__(
        self, qrel: Mapping[str, int], run: Mapping[str, float], /
    ) -> float | None:
        """Computes the metric of the retrieved cases against the relevance judgments, None if it is undefined."""
        ...


class Storage[K, V](Index[K, V], Protocol):
    """A storage of cases that allocates the keys of new cases, such as those that `cbrkit.retain` stores.

    Allocating the keys, checking the expected and inserted cases, and writing the cases is one mutation
    under the lock or transaction of the storage,
    so concurrent writers never receive the same key or overwrite each other's changes.
    """

    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
        add: Sequence[V] = (),
        *,
        insert: Casebase[K, V] | None = None,
        expected: Casebase[K, V] | None = None,
    ) -> Sequence[K]:
        """Insert, replace, delete, and add cases as one mutation.

        Args:
            upsert: Cases to insert or replace under their keys.
            delete: Keys of the cases to delete.
            add: Cases to add under new keys.
            insert: Cases to add under their keys, which no stored case may have.
            expected: The cases that the mutation was planned on, unchecked if None.

        Returns:
            The keys of the added cases in their order, by default
            the integers after the largest key if the keys are integers, new UUIDv7 strings otherwise.

        Raises:
            ValueError: If a key is both upserted and inserted, or written and deleted.
            cbrkit.errors.Conflict: If the storage holds other cases than the expected ones
                or a case under a key to insert.
        """
        ...
