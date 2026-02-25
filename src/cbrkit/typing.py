from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "AdaptationFunc",
    "AggregatorFunc",
    "AnyAdaptationFunc",
    "AnyConversionFunc",
    "AnyNamedFunc",
    "AnyPoolingFunc",
    "AnyPositionalFunc",
    "AnySimFunc",
    "InternalFunc",
    "BatchAdaptationFunc",
    "BatchConversionFunc",
    "BatchNamedFunc",
    "BatchPoolingFunc",
    "BatchPositionalFunc",
    "BatchSimFunc",
    "Casebase",
    "CbrFunc",
    "ConversionFunc",
    "EvalMetricFunc",
    "Factory",
    "FilePath",
    "Float",
    "HasMetadata",
    "IndexableFunc",
    "JsonDict",
    "JsonEntry",
    "MapAdaptationFunc",
    "MaybeFactory",
    "MaybeFactories",
    "MaybeSequence",
    "NamedFunc",
    "NumpyArray",
    "PoolingFunc",
    "PositionalFunc",
    "QueryCaseMatrix",
    "ReduceAdaptationFunc",
    "SimpleAdaptationFunc",
    "ComplexAdaptationFunc",
    "RetainerFunc",
    "RetrieverFunc",
    "ReuserFunc",
    "ReviserFunc",
    "SimFunc",
    "SimMap",
    "SimSeq",
    "SparseVector",
    "StructuredValue",
    "SynthesizerFunc",
    "SynthesizerPromptFunc",
    "Value",
    "WrappedObject",
]

type JsonEntry = (
    Mapping[str, "JsonEntry"] | Sequence["JsonEntry"] | str | int | float | bool | None
)
type JsonDict = dict[str, JsonEntry]
type NumpyArray = npt.NDArray[np.float64]
type SparseVector = dict[int, float]


@dataclass(slots=True, frozen=True)
class StructuredValue[T]:
    value: T


class WrappedObject[T](ABC):
    __wrapped__: T


class HasMetadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> JsonDict: ...


class InternalFunc(ABC):
    """Marker for internal functions excluded from result steps."""

    pass


class IndexableFunc[T, K = T](Protocol):
    """Supports pre-indexing data for efficient processing.

    Type Parameters:
        T: The full index type (e.g., `Casebase[K, V]`).
        K: The keys/delete type (e.g., `Collection[K]`).
    """

    @property
    def index(self) -> T: ...

    def has_index(self) -> bool: ...

    def create_index(self, data: T, /) -> None: ...

    def update_index(self, data: T, /) -> None: ...

    def delete_index(self, keys: K, /) -> None: ...


type Value[T] = T | StructuredValue[T]
type Float = Value[float]
type FilePath = str | Path
type Casebase[K, V] = Mapping[K, V]
type SimMap[K, S: Float = float] = Mapping[K, S]
type SimSeq[S: Float = float] = Sequence[S]
type QueryCaseMatrix[Q, C, V] = Mapping[Q, Mapping[C, V]]
type Factory[T] = Callable[[], T]
type MaybeFactory[T] = T | Factory[T]
type MaybeSequence[T] = T | Sequence[T]
type MaybeFactories[T] = T | Factory[T] | Sequence[T | Factory[T]]


class ConversionFunc[U, V](Protocol):
    """Converts a single value from type U to type V."""

    def __call__(
        self,
        batch: U,
        /,
    ) -> V: ...


class BatchConversionFunc[U, V](Protocol):
    """Converts multiple values from type U to type V in batch."""

    def __call__(
        self,
        batches: Sequence[U],
        /,
    ) -> Sequence[V]: ...


type AnyConversionFunc[U, V] = ConversionFunc[U, V] | BatchConversionFunc[U, V]


class PositionalFunc[T](Protocol):
    """Callable that accepts any positional and keyword arguments and returns T."""

    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...


class BatchPositionalFunc[T](Protocol):
    """Processes multiple inputs in batch, returning a sequence of T."""

    def __call__(
        self,
        batches: Sequence[Any],
        /,
    ) -> Sequence[T]: ...


type AnyPositionalFunc[T] = PositionalFunc[T] | BatchPositionalFunc[T]


class NamedFunc[T](Protocol):
    """Generic callable that returns type T."""

    def __call__(self, *args: Any, **kwargs: Any) -> T: ...


class BatchNamedFunc[T](Protocol):
    """Processes multiple generic inputs in batch, returning a sequence of T."""

    def __call__(
        self,
        batches: Sequence[Any],
        /,
    ) -> Sequence[T]: ...


type AnyNamedFunc[T] = NamedFunc[T] | BatchNamedFunc[T]


class SimFunc[V, S: Float = float](Protocol):
    """Computes similarity between two values of type V."""

    def __call__(
        self,
        x: V,
        y: V,
        /,
    ) -> S: ...


class BatchSimFunc[V, S: Float = float](Protocol):
    """Computes similarities for multiple value pairs in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> SimSeq[S]: ...


type AnySimFunc[V, S: Float = float] = SimFunc[V, S] | BatchSimFunc[V, S]


class CbrFunc[K, V, S: Float = float](Protocol):
    """Unified protocol for all CBR cycle phases.

    Each phase function takes a sequence of (casebase, query) batches
    and returns a sequence of (casebase, score_map) results.
    The casebase in the output may differ from the input depending on the
    phase (e.g., adapted cases in reuse, newly stored cases in retain).
    The score map assigns a floating-point score to each case in the output
    casebase, with phase-specific semantics.
    """

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]: ...


class RetrieverFunc[K, V, S: Float = float](CbrFunc[K, V, S], Protocol):
    """Retrieves similar cases from casebases for given queries."""

    ...


class AdaptationFunc[V](Protocol):
    """Adapts a case to match a query."""

    def __call__(
        self,
        case: V,
        query: V,
        /,
    ) -> V: ...


class BatchAdaptationFunc[V](Protocol):
    """Adapts multiple case-query pairs in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> Sequence[V]: ...


class MapAdaptationFunc[K, V](Protocol):
    """Adapts all cases in a casebase to match a query."""

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> Casebase[K, V]: ...


class ReduceAdaptationFunc[K, V](Protocol):
    """Selects and adapts the best case from a casebase for a query."""

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> tuple[K, V]: ...


type SimpleAdaptationFunc[V] = AdaptationFunc[V] | BatchAdaptationFunc[V]

type ComplexAdaptationFunc[K, V] = MapAdaptationFunc[K, V] | ReduceAdaptationFunc[K, V]

type AnyAdaptationFunc[K, V] = SimpleAdaptationFunc[V] | ComplexAdaptationFunc[K, V]


class ReuserFunc[K, V, S: Float = float](CbrFunc[K, V, S], Protocol):
    """Reuses cases by adapting and computing similarities for queries."""

    ...


class ReviserFunc[K, V, S: Float = float](CbrFunc[K, V, S], Protocol):
    """Revises solutions by assessing quality and optionally repairing them."""

    ...


class RetainerFunc[K, V, S: Float = float](CbrFunc[K, V, S], Protocol):
    """Retains cases in the casebase."""

    ...


class AggregatorFunc[K, S: Float = float](Protocol):
    """Aggregates multiple similarity scores into a single value."""

    def __call__(
        self,
        similarities: SimMap[K, S] | SimSeq[S],
        /,
    ) -> float: ...


class PoolingFunc[T](Protocol):
    """Pools multiple values into a single representative value."""

    def __call__(
        self,
        values: Sequence[T],
        /,
    ) -> T: ...


class BatchPoolingFunc[T](Protocol):
    """Pools multiple sequences of values in batch."""

    def __call__(
        self,
        batches: Sequence[Sequence[T]],
        /,
    ) -> Sequence[T]: ...


type AnyPoolingFunc[T] = PoolingFunc[T] | BatchPoolingFunc[T]


class EvalMetricFunc(Protocol):
    """Evaluates retrieval quality using relevance judgments."""

    def __call__(
        self,
        /,
        qrels: Mapping[str, Mapping[str, int]],
        run: Mapping[str, Mapping[str, float]],
        k: int,
        relevance_level: int,
    ) -> float: ...


class SynthesizerPromptFunc[T, K, V, S: Float = float](Protocol):
    """Creates prompts from casebases, queries, and similarities."""

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V | None,
        similarities: SimMap[K, S] | None,
        /,
    ) -> T: ...


class SynthesizerFunc[T, K, V, S: Float = float](Protocol):
    """Synthesizes results from casebases, queries, and similarities in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]],
    ) -> Sequence[T]: ...
