from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable


@runtime_checkable
class FloatProtocol(Protocol):
    value: float


AnyFloat = float | FloatProtocol

FilePath = str | Path
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")
ValueType_contra = TypeVar("ValueType_contra", contravariant=True)
ValueType_cov = TypeVar("ValueType_cov", covariant=True)
Casebase = Mapping[KeyType, ValueType]

SimType = TypeVar("SimType", bound=AnyFloat)
SimType_cov = TypeVar("SimType_cov", bound=AnyFloat, covariant=True)
SimType_contra = TypeVar("SimType_contra", bound=AnyFloat, contravariant=True)

SimMap = Mapping[KeyType, SimType]
SimSeq = Sequence[SimType]
SimSeqOrMap = SimMap[KeyType, SimType] | SimSeq[SimType]


# Parameter names must match so that the signature can be inspected, do not add `/` here!
class SimMapFunc(Protocol[KeyType, ValueType_contra, SimType_cov]):
    def __call__(
        self, x_map: Mapping[KeyType, ValueType_contra], y: ValueType_contra
    ) -> SimMap[KeyType, SimType_cov]: ...


class SimSeqFunc(Protocol[ValueType_contra, SimType_cov]):
    def __call__(
        self, pairs: Sequence[tuple[ValueType_contra, ValueType_contra]], /
    ) -> SimSeq[SimType_cov]: ...


class SimPairFunc(Protocol[ValueType_contra, SimType_cov]):
    def __call__(self, x: ValueType_contra, y: ValueType_contra, /) -> SimType_cov: ...


AnySimFunc = (
    SimMapFunc[KeyType, ValueType, SimType]
    | SimSeqFunc[ValueType, SimType]
    | SimPairFunc[ValueType, SimType]
)


class AggregatorFunc(Protocol[KeyType, SimType_contra]):
    def __call__(
        self,
        similarities: SimSeqOrMap[KeyType, SimType_contra],
        /,
    ) -> float: ...


class PoolingFunc(Protocol):
    def __call__(
        self,
        similarities: SimSeq[float],
        /,
    ) -> float: ...
