from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
)

FilePath = str | Path

CaseType = TypeVar("CaseType", contravariant=True)
CaseName = Any
Casebase = Mapping[CaseName, CaseType]

DataType = TypeVar("DataType", contravariant=True)

SimilarityValue = float
DataSimilarityMap = Mapping[DataType, SimilarityValue]
CaseSimilarityMap = Mapping[CaseName, SimilarityValue]
SimilaritySequence = Sequence[SimilarityValue]
SimilarityValues = TypeVar(
    "SimilarityValues",
    SimilaritySequence,
    CaseSimilarityMap,
    DataSimilarityMap,
    contravariant=True,
)


class AggregateFunc(Protocol[SimilarityValues]):
    def __call__(self, similarities: SimilarityValues) -> SimilarityValue:
        ...


class CaseSimilarityBatchFunc(Protocol[CaseType]):
    def __call__(
        self, casebase: Casebase[CaseType], query: CaseType
    ) -> CaseSimilarityMap:
        ...


class CaseSimilaritySingleFunc(Protocol[CaseType]):
    def __call__(self, case: CaseType, query: CaseType) -> SimilarityValue:
        ...


class DataSimilarityBatchFunc(Protocol[DataType]):
    def __call__(self, *args: tuple[DataType, DataType]) -> SimilaritySequence:
        ...


class DataSimilaritySingleFunc(Protocol[DataType]):
    def __call__(self, x: DataType, y: DataType) -> SimilarityValue:
        ...


Pooling = Literal[
    "mean",
    "fmean",
    "geometric_mean",
    "harmonic_mean",
    "median",
    "median_low",
    "median_high",
    "mode",
    "min",
    "max",
    "sum",
]


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: CaseSimilarityMap
    ranking: list[CaseName]
    casebase: Casebase[CaseType]


class Retriever(Protocol[CaseType]):
    def __call__(
        self,
        casebase: Casebase[CaseType],
        query: CaseType,
    ) -> RetrievalResult[CaseType]:
        ...


# TODO: Create helper for astar search
# TODO: Rest API
# TODO: CLI Interface for two-dimensional (i.e., tabular) data
# TODO: How to handle similarity function composition via config files (e.g., for CLI)?
