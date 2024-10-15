import dataclasses
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


@runtime_checkable
class SupportsMetadata(Protocol):
    @property
    def metadata(self) -> JsonDict:
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)

        return {}


class SimMapFunc[K, V, S: Float](Protocol):
    def __call__(self, x_map: Mapping[K, V], y: V, /) -> SimMap[K, S]: ...


class SimSeqFunc[V, S: Float](Protocol):
    def __call__(self, pairs: Sequence[tuple[V, V]], /) -> SimSeq[S]: ...


class SimPairFunc[V, S: Float](Protocol):
    def __call__(self, x: V, y: V, /) -> S: ...


type AnySimFunc[V, S: Float] = SimPairFunc[V, S] | SimSeqFunc[V, S]


class RetrieverFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        casebase: Mapping[K, V],
        query: V,
        processes: int,
    ) -> SimMap[K, S]: ...


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


class AdaptationFunc[K, V, S: Float](Protocol):
    def __call__(
        self,
        x_map: Casebase[K, V],
        y: V,
        sim_func: AnySimFunc[V, S],
    ) -> V: ...
