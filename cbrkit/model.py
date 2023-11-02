from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Generic,
    Hashable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
)

FilePath = str | Path
SimilarityValue = float

CaseType = TypeVar("CaseType")
CaseName = Hashable
Casebase = Mapping[CaseName, CaseType]

SimilarityType = Literal["equality"]
SimilarityFunc = Callable[[CaseType, CaseType], SimilarityValue]

LoadFormat = Literal[".csv", ".yaml", ".yml", ".json", ".toml"]
LoadFunc = Callable[[FilePath], Casebase[CaseType]]

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

# TODO: Create helper for astar search
# TODO: Rest API
# TODO: CLI Interface for two-dimensional (i.e., tabular) data


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: dict[CaseName, SimilarityValue]
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
