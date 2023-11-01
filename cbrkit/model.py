from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Hashable, Literal, TypeVar

FilePath = str | Path
SimilarityValue = float

CaseType = TypeVar("CaseType")
CaseName = Hashable
Casebase = dict[CaseName, CaseType]

SimilarityType = Literal["equality"]
SimilarityFunc = Callable[[CaseType, CaseType], SimilarityValue]

LoadFormat = Literal["csv", "yaml", "yml", "json", "toml"]
LoadFunc = Callable[[FilePath], Casebase[CaseType]]

RetrievalType = Literal["linear"]
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

# TODO: How to perform astar search?
# TODO: Is the linear retriever the only option?
# TODO: Rest API
# TODO: CLI Interface for two-dimensional (i.e., tabular) data


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: dict[CaseName, SimilarityValue]
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
