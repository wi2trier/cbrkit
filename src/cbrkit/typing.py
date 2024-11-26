from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class AnnotatedFloat(Protocol):
    value: float


type JsonEntry = (
    Mapping[str, "JsonEntry"] | Sequence["JsonEntry"] | str | int | float | bool | None
)
type JsonDict = dict[str, JsonEntry]
type Float = float | AnnotatedFloat
type FilePath = str | Path
type Casebase[K, V] = Mapping[K, V]
type SimMap[K, S: Float] = Mapping[K, S]
type SimSeq[S: Float] = Sequence[S]
type SimSeqOrMap[K, S: Float] = SimMap[K, S] | SimSeq[S]
type QueryCaseMatrix[Q, C, V] = Mapping[Q, Mapping[C, V]]


@runtime_checkable
class HasMetadata(Protocol):
    @property
    def metadata(self) -> JsonDict: ...


class SimMapFunc[K, V, S: Float](Protocol):
    def __call__(self, x_map: Mapping[K, V], y: V, /) -> SimMap[K, S]: ...


class SimSeqFunc[V, S: Float](Protocol):
    def __call__(self, pairs: Sequence[tuple[V, V]], /) -> SimSeq[S]: ...


class SimPairFunc[V, S: Float](Protocol):
    def __call__(self, x: V, y: V, /) -> S: ...


type AnySimFunc[V, S: Float] = SimPairFunc[V, S] | SimSeqFunc[V, S]


class RetrieverFunc[K, V, S: Float](Protocol):
    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V]]
    ) -> Sequence[SimMap[K, S]]: ...


class AdaptPairFunc[V](Protocol):
    def __call__(
        self,
        case: V,
        query: V,
    ) -> V: ...


class AdaptMapFunc[K, V](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]: ...


class AdaptReduceFunc[K, V](Protocol):
    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> tuple[K, V]: ...


type AnyAdaptFunc[K, V] = AdaptPairFunc[V] | AdaptMapFunc[K, V] | AdaptReduceFunc[K, V]


class ReuserFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]],
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]: ...


class AggregatorFunc[K, S: Float](Protocol):
    def __call__(
        self,
        similarities: SimSeqOrMap[K, S],
        /,
    ) -> float: ...


class PoolingFunc(Protocol):
    def __call__(
        self,
        similarities: SimSeq[float],
        /,
    ) -> float: ...


class EvalMetricFunc(Protocol):
    def __call__(
        self,
        qrels: Mapping[str, Mapping[str, int]],
        run: Mapping[str, Mapping[str, float]],
        k: int,
        relevance_level: int,
    ) -> float: ...
