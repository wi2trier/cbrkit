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

SimilarityValue = float
SimilarityMap = Mapping[KeyType, SimilarityValue]
SimilaritySequence = Sequence[SimilarityValue]
SimilarityValues = TypeVar(
    "SimilarityValues",
    SimilaritySequence,
    SimilarityMap,
)
SimilarityValues_contra = TypeVar(
    "SimilarityValues_contra",
    SimilaritySequence,
    SimilarityMap,
    contravariant=True,
)


class SimMapFunc(Protocol[KeyType, ValueType_contra]):
    def __call__(
        self, x_map: Mapping[KeyType, ValueType_contra], y: ValueType_contra, /
    ) -> SimilarityMap[KeyType]:
        ...


class SimSeqFunc(Protocol[ValueType_contra]):
    def __call__(
        self, pairs: Sequence[tuple[ValueType_contra, ValueType_contra]], /
    ) -> SimilaritySequence:
        ...


# Parameter names must match so that the signature can be inspected, do not add `/` here!
class SimFunc(Protocol[ValueType_contra]):
    def __call__(self, x: ValueType_contra, y: ValueType_contra) -> SimilarityValue:
        ...


SimPairOrSeqFunc = SimFunc[ValueType] | SimSeqFunc[ValueType]
SimPairOrMapFunc = SimFunc[ValueType] | SimMapFunc[KeyType, ValueType]


class RetrievalResultProtocol(Protocol[KeyType, ValueType]):
    similarities: SimilarityMap[KeyType]
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


class AggregateFunc(Protocol[SimilarityValues_contra]):
    def __call__(self, similarities: SimilarityValues_contra, /) -> SimilarityValue:
        ...
