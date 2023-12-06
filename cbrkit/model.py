from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generic,
    Literal,
    TypeVar,
)

FilePath = str | Path

CaseType = TypeVar("CaseType")
CaseName = str
Casebase = Mapping[CaseName, CaseType]

SimilarityValue = float
SimilarityMap = Mapping[CaseName, SimilarityValue]

SimilarityFuncName = Literal["equality"]
SimilarityBatchFunc = Callable[[Casebase[CaseType], CaseType], SimilarityMap]
SimilaritySingleFunc = Callable[[CaseType, CaseType], SimilarityValue]

AggregationOperation = Literal[
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
AggregationType = TypeVar(
    "AggregationType",
    Sequence[SimilarityValue],
    Mapping[Hashable, SimilarityValue],
)

# TODO: Create helper for astar search
# TODO: Rest API
# TODO: CLI Interface for two-dimensional (i.e., tabular) data
# TODO: How to design similarity helper methods


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: SimilarityMap
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
