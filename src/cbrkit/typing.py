from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Protocol,
)


class StructuredValue[T](ABC):
    value: T


type JsonEntry = (
    Mapping[str, "JsonEntry"] | Sequence["JsonEntry"] | str | int | float | bool | None
)
type JsonDict = dict[str, JsonEntry]
type Float = float | StructuredValue[float]
type Prompt[P] = P | StructuredValue[P]
type FilePath = str | Path
type Casebase[K, V] = Mapping[K, V]
type SimMap[K, S: Float] = Mapping[K, S]
type SimSeq[S: Float] = Sequence[S]
type QueryCaseMatrix[Q, C, V] = Mapping[Q, Mapping[C, V]]


class HasMetadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> JsonDict: ...


class ConversionFunc[U, V](Protocol):
    def __call__(
        self,
        obj: U,
        /,
    ) -> V: ...


class PositionalFunc[T](Protocol):
    def __call__(
        self,
        *args: Any,
    ) -> T: ...


class BatchPositionalFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Any, ...]],
    ) -> Sequence[T]: ...


type AnyPositionalFunc[T] = PositionalFunc[T] | BatchPositionalFunc[T]


class NamedFunc[T](Protocol):
    def __call__(self, **kwargs: Any) -> T: ...


class NamedBatchFunc[T](Protocol):
    def __call__(
        self,
        batches: Sequence[dict[str, Any]],
    ) -> Sequence[T]: ...


type AnyNamedFunc[T] = NamedFunc[T] | NamedBatchFunc[T]


class SimFunc[V, S: Float](Protocol):
    def __call__(
        self,
        x: V,
        y: V,
    ) -> S: ...


class BatchSimFunc[V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
    ) -> SimSeq[S]: ...


type AnySimFunc[V, S: Float] = SimFunc[V, S] | BatchSimFunc[V, S]


class RetrieverFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[SimMap[K, S]]: ...


class AdaptationFunc[V](Protocol):
    def __call__(
        self,
        case: V,
        query: V,
    ) -> V: ...


class BatchAdaptationFunc[V](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[V, V]],
    ) -> Sequence[V]: ...


class AdaptationMapFunc[K, V](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]: ...


class AdaptationReduceFunc[K, V](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> tuple[K, V]: ...


type AnyAdaptationFunc[K, V] = (
    AdaptationFunc[V] | AdaptationMapFunc[K, V] | AdaptationReduceFunc[K, V]
)


class ReuserFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
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


class EvalMetricFunc(Protocol):
    def __call__(
        self,
        qrels: Mapping[str, Mapping[str, int]],
        run: Mapping[str, Mapping[str, float]],
        k: int,
        relevance_level: int,
    ) -> float: ...


class GenerationFunc[P, R](Protocol):
    def __call__(
        self,
        prompt: P,
    ) -> R: ...


class BatchGenerationFunc[P, R](Protocol):
    def __call__(
        self,
        batches: Sequence[P],
    ) -> Sequence[R]: ...


type AnyGenerationFunc[P, R] = GenerationFunc[P, R] | BatchGenerationFunc[P, R]


class PromptFunc[T, K, V, S: Float](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
        similarities: SimMap[K, S] | None,
    ) -> T: ...


class PoolingPromptFunc[P, V](Protocol):
    def __call__(
        self,
        values: Sequence[V],
        /,
    ) -> P: ...


class RagFunc[T, K, V, S: Float](Protocol):
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[T]: ...
