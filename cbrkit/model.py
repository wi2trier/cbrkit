from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Generic,
    Literal,
    TypeVar,
)

FilePath = str | Path
SimilarityValue = float

CaseType = TypeVar("CaseType")
CaseName = Hashable
Casebase = Mapping[CaseName, CaseType]

SimilarityType = Literal["equality"]
SimilarityFunc = Callable[[CaseType, CaseType], SimilarityValue]

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


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: dict[CaseName, SimilarityValue]
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
