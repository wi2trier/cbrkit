from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeVar,
)

FilePath = str | Path

CaseType = TypeVar("CaseType")
CaseName = str
Casebase = Mapping[CaseName, CaseType]

DataType = TypeVar("DataType", contravariant=True)

SimilarityValue = float
DataSimilarityMap = Mapping[DataType, SimilarityValue]
CaseSimilarityMap = Mapping[CaseName, SimilarityValue]
SimilaritySequence = Sequence[SimilarityValue]
SimilarityValues = TypeVar(
    "SimilarityValues", SimilaritySequence, CaseSimilarityMap, DataSimilarityMap
)

SimilarityFuncName = Literal["equality"]
CaseSimilarityBatchFunc = Callable[[Casebase[CaseType], CaseType], CaseSimilarityMap]
CaseSimilaritySingleFunc = Callable[[CaseType, CaseType], SimilarityValue]

# DataType = TypeVar("DataType")
# DataSimilaritySingleFunc = Callable[[DataType, DataType], SimilarityValue]
# DataSimilarityBatchFunc = Callable[[Sequence[tuple[DataType, DataType]]], SimilarityValue]


# class SimilarityBatchFunc(Protocol[CaseType]):
#     def __call__(self, casebase: Casebase[CaseType], query: CaseType) -> SimilarityCaseMap:
#         ...


# class SimilaritySingleFunc(Protocol[CaseType]):
#     def __call__(self, case: CaseType, query: CaseType) -> SimilarityValue:
#         ...


class DataSimilarityBatchFunc(Protocol[DataType]):
    def __call__(self, *args: tuple[DataType, DataType]) -> SimilaritySequence:
        ...


class DataSimilaritySingleFunc(Protocol[DataType]):
    def __call__(self, data1: DataType, data2: DataType) -> SimilarityValue:
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

# TODO: Create helper for astar search
# TODO: Rest API
# TODO: CLI Interface for two-dimensional (i.e., tabular) data
# TODO: How to design similarity helper methods


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: CaseSimilarityMap
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
