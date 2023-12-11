from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Protocol,
    TypeVar,
)

FilePath = str | Path
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")
ValueType_contra = TypeVar("ValueType_contra", contravariant=True)
Casebase = Mapping[KeyType, ValueType]

SimVal = float
SimMap = Mapping[KeyType, SimVal]
SimSeq = Sequence[SimVal]
SimSeqOrMap = SimMap[KeyType] | SimSeq


class SimMapFunc(Protocol[KeyType, ValueType_contra]):
    def __call__(
        self, x_map: Mapping[KeyType, ValueType_contra], y: ValueType_contra, /
    ) -> SimMap[KeyType]:
        ...


class SimSeqFunc(Protocol[ValueType_contra]):
    def __call__(
        self, pairs: Sequence[tuple[ValueType_contra, ValueType_contra]], /
    ) -> SimSeq:
        ...


# Parameter names must match so that the signature can be inspected, do not add `/` here!
class SimPairFunc(Protocol[ValueType_contra]):
    def __call__(self, x: ValueType_contra, y: ValueType_contra) -> SimVal:
        ...


AnySimFunc = (
    SimMapFunc[KeyType, ValueType] | SimSeqFunc[ValueType] | SimPairFunc[ValueType]
)


class RetrievalResultProtocol(Protocol[KeyType, ValueType]):
    similarities: SimMap[KeyType]
    ranking: list[KeyType]
    casebase: Casebase[KeyType, ValueType]


class RetrieveFunc(Protocol[KeyType, ValueType]):
    def __call__(
        self,
        casebase: Casebase[KeyType, ValueType],
        query: ValueType,
        /,
    ) -> RetrievalResultProtocol[KeyType, ValueType]:
        ...


class AggregatorFunc(Protocol[KeyType]):
    def __call__(
        self,
        similarities: SimSeqOrMap[KeyType],
        /,
    ) -> SimVal:
        ...


class PoolingFunc(Protocol):
    def __call__(
        self,
        similarities: SimSeq,
        /,
    ) -> SimVal:
        ...
