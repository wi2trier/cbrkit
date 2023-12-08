from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Protocol,
    TypeVar,
)

FilePath = str | Path

CaseName = TypeVar("CaseName")
CaseType = TypeVar("CaseType")
CaseType_contra = TypeVar("CaseType_contra", contravariant=True)
Casebase = Mapping[CaseName, CaseType]
DataType = TypeVar("DataType", contravariant=True)
DataType_contra = TypeVar("DataType_contra", contravariant=True)

SimilarityValue = float
SimilarityKey = TypeVar("SimilarityKey")
SimilarityMap = Mapping[SimilarityKey, SimilarityValue]
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


class CaseSimBatchFunc(Protocol[CaseName, CaseType_contra]):
    def __call__(
        self, casebase: Casebase[CaseName, CaseType_contra], query: CaseType_contra
    ) -> SimilarityMap[CaseName]:
        ...


class CaseSimFunc(Protocol[CaseType_contra]):
    def __call__(
        self, case: CaseType_contra, query: CaseType_contra
    ) -> SimilarityValue:
        ...


class DataSimBatchFunc(Protocol[DataType_contra]):
    def __call__(
        self, *args: tuple[DataType_contra, DataType_contra]
    ) -> SimilaritySequence:
        ...


class DataSimFunc(Protocol[DataType_contra]):
    def __call__(self, x: DataType_contra, y: DataType_contra) -> SimilarityValue:
        ...


class RetrievalResultProtocol(Protocol[CaseName, CaseType]):
    similarities: SimilarityMap[CaseName]
    ranking: list[CaseName]
    casebase: Casebase[CaseName, CaseType]


class RetrieveFunc(Protocol[CaseName, CaseType]):
    def __call__(
        self,
        casebase: Casebase[CaseName, CaseType],
        query: CaseType,
    ) -> RetrievalResultProtocol[CaseName, CaseType]:
        ...


class AggregateFunc(Protocol[SimilarityValues_contra]):
    def __call__(self, similarities: SimilarityValues_contra) -> SimilarityValue:
        ...
