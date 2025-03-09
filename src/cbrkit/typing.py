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
    def __call__(
        self,
        batch: U,
        /,
    ) -> V: ...


class BatchConversionFunc[U, V](Protocol):
    def __call__(
        self,
        batches: Sequence[U],
        /,
    ) -> Sequence[V]: ...


type AnyConversionFunc[U, V] = ConversionFunc[U, V] | BatchConversionFunc[U, V]


class PositionalFunc[T](Protocol):
    def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> T: ...


class BatchPositionalFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[Any],
        /,
    ) -> Sequence[T]: ...


type AnyPositionalFunc[T] = PositionalFunc[T] | BatchPositionalFunc[T]


class NamedFunc[T](Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> T: ...


class BatchNamedFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[Any],
        /,
    ) -> Sequence[T]: ...


type AnyNamedFunc[T] = NamedFunc[T] | BatchNamedFunc[T]


class SimFunc[V, S: Float](Protocol):
    def __call__(
        self,
        x: V,
        y: V,
        /,
    ) -> S: ...


class BatchSimFunc[V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> SimSeq[S]: ...


type AnySimFunc[V, S: Float] = SimFunc[V, S] | BatchSimFunc[V, S]


class RetrieverFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[SimMap[K, S]]: ...


class AdaptationFunc[V](Protocol):
    def __call__(
        self,
        case: V,
        query: V,
        /,
    ) -> V: ...


class BatchAdaptationFunc[V](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
        /,
    ) -> Sequence[V]: ...


class MapAdaptationFunc[K, V](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        /,
    ) -> Casebase[K, V]: ...


class ReduceAdaptationFunc[K, V](Protocol):
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
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]: ...


class AggregatorFunc[K, S: Float](Protocol):
    def __call__(
        self,
        similarities: SimMap[K, S] | SimSeq[S],
        /,
    ) -> float: ...


class PoolingFunc[T](Protocol):
    def __call__(
        self,
        values: Sequence[T],
        /,
    ) -> T: ...


class BatchPoolingFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[Sequence[T]],
        /,
    ) -> Sequence[T]: ...


type AnyPoolingFunc[T] = PoolingFunc[T] | BatchPoolingFunc[T]


class EvalMetricFunc(Protocol):
    def __call__(
        self,
        /,
        qrels: Mapping[str, Mapping[str, int]],
        run: Mapping[str, Mapping[str, float]],
        k: int,
        relevance_level: int,
    ) -> float: ...


class SynthesizerPromptFunc[T, K, V, S: Float](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V | None,
        similarities: SimMap[K, S] | None,
        /,
    ) -> T: ...


class SynthesizerFunc[T, K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]],
    ) -> Sequence[T]: ...
