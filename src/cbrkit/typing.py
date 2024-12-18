from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

__all__ = [
    "JsonEntry",
    "JsonDict",
    "NumpyArray",
    "AdaptationFunc",
    "AggregatorFunc",
    "AnyAdaptationFunc",
    "AnyConversionFunc",
    "AnyNamedFunc",
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
    "ConversionPoolingFunc",
    "EvalMetricFunc",
    "FilePath",
    "Float",
    "HasMetadata",
    "MapAdaptationFunc",
    "NamedFunc",
    "PoolingFunc",
    "PositionalFunc",
    "Prompt",
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
    "KeyValueStore",
    "WrappedObject",
]

type JsonEntry = (
    Mapping[str, "JsonEntry"] | Sequence["JsonEntry"] | str | int | float | bool | None
)
type JsonDict = dict[str, JsonEntry]
type NumpyArray = npt.NDArray[np.float_]


class StructuredValue[T](ABC):
    value: T


class WrappedObject[T](ABC):
    __wrapped__: T


class HasMetadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> JsonDict: ...


type Float = float | StructuredValue[float]
type Prompt[P] = P | StructuredValue[P]
type FilePath = str | Path
type Casebase[K, V] = Mapping[K, V]
type SimMap[K, S: Float] = Mapping[K, S]
type SimSeq[S: Float] = Sequence[S]
type QueryCaseMatrix[Q, C, V] = Mapping[Q, Mapping[C, V]]


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
    ) -> T: ...


class BatchPositionalFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Any, ...]],
        /,
    ) -> Sequence[T]: ...


type AnyPositionalFunc[T] = PositionalFunc[T] | BatchPositionalFunc[T]


class NamedFunc[T](Protocol):
    def __call__(self, **kwargs: Any) -> T: ...


class BatchNamedFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[dict[str, Any]],
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


class ConversionPoolingFunc[U, V](Protocol):
    def __call__(
        self,
        values: Sequence[U],
        /,
    ) -> V: ...


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
        query: V,
        similarities: SimMap[K, S] | None,
        /,
    ) -> T: ...


class SynthesizerFunc[T, K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[T]: ...


class KeyValueStore[K, V](BatchConversionFunc[K, V], Protocol):
    func: BatchConversionFunc[K, V]
    store: MutableMapping[K, V]
    path: FilePath | None
    frozen: bool

    def dump(self) -> None: ...

    def __call__(
        self,
        batches: Sequence[K],
        /,
    ) -> Sequence[V]: ...
