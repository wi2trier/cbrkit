from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Protocol,
    TypeVar,
)

FilePath = str | Path
OuterKeyType = TypeVar("OuterKeyType")
KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")
ValueType_contra = TypeVar("ValueType_contra", contravariant=True)
Casebase = Mapping[KeyType, ValueType]

SimType = float
SimMap = Mapping[KeyType, SimType]
SimSeq = Sequence[SimType]
SimVals = SimMap[KeyType] | SimSeq


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
class SimFunc(Protocol[ValueType_contra]):
    def __call__(self, x: ValueType_contra, y: ValueType_contra) -> SimType:
        ...


SimPairOrSeqFunc = SimSeqFunc[ValueType] | SimFunc[ValueType]
SimPairOrMapFunc = SimMapFunc[KeyType, ValueType] | SimFunc[ValueType]


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


class AggregateFunc(Protocol[KeyType]):
    def __call__(
        self,
        similarities: SimVals[KeyType],
        /,
    ) -> SimType:
        ...


class AggregateMapFunc(Protocol[KeyType, OuterKeyType]):
    def __call__(
        self,
        similarities: Mapping[OuterKeyType, SimVals[KeyType]],
        /,
    ) -> Mapping[OuterKeyType, SimType]:
        ...
