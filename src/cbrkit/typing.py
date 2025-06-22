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
    "BatchAdaptationFunc",
    "BatchConversionFunc",
    "BatchNamedFunc",
    "BatchPoolingFunc",
    "BatchPositionalFunc",
    "BatchSimFunc",
    "Casebase",
    "ConversionFunc",
    "EvalMetricFunc",
    "Factory",
    "FilePath",
    "Float",
    "HasMetadata",
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
    "RetrieverFunc",
    "ReuserFunc",
    "SimFunc",
    "SimMap",
    "SimSeq",
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


@dataclass(slots=True, frozen=True)
class StructuredValue[T]:
    value: T


class WrappedObject[T](ABC):
    __wrapped__: T


class HasMetadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> JsonDict: ...


type Value[T] = T | StructuredValue[T]
type Float = Value[float]
type FilePath = str | Path
type Casebase[K, V] = Mapping[K, V]
type SimMap[K, S: Float] = Mapping[K, S]
type SimSeq[S: Float] = Sequence[S]
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


class SimFunc[V, S: Float](Protocol):
    """Computes similarity between two values of type V."""

    def __call__(
        self,
        x: V,
        y: V,
        /,
    ) -> S: ...


class BatchSimFunc[V, S: Float](Protocol):
    """Computes similarities for multiple value pairs in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> SimSeq[S]: ...


type AnySimFunc[V, S: Float] = SimFunc[V, S] | BatchSimFunc[V, S]


class RetrieverFunc[K, V, S: Float](Protocol):
    """Retrieves similar cases from casebases for given queries."""

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[SimMap[K, S]]: ...


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


type AnyAdaptationFunc[K, V] = (
    AdaptationFunc[V] | MapAdaptationFunc[K, V] | ReduceAdaptationFunc[K, V]
)


class ReuserFunc[K, V, S: Float](Protocol):
    """Reuses cases by adapting and computing similarities for queries."""

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]: ...


class AggregatorFunc[K, S: Float](Protocol):
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


class SynthesizerPromptFunc[T, K, V, S: Float](Protocol):
    """Creates prompts from casebases, queries, and similarities."""

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V | None,
        similarities: SimMap[K, S] | None,
        /,
    ) -> T: ...


class SynthesizerFunc[T, K, V, S: Float](Protocol):
    """Synthesizes results from casebases, queries, and similarities in batch."""

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]],
    ) -> Sequence[T]: ...
