from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Hashable, Literal, TypeVar

FilePath = str | Path

CaseType = TypeVar("CaseType")
CaseName = Hashable
Casebase = dict[CaseName, CaseType]

SimilarityType = Literal["equality"]
SimilarityFunc = Callable[[CaseType, CaseType], float]

LoadFormat = Literal["csv", "yaml", "yml", "json", "toml"]
LoadFunc = Callable[[FilePath], Casebase[CaseType]]

RetrievalType = Literal["linear"]

# TODO: Expand model to support yaml and csv data sources


@dataclass
class RetrievalResult(Generic[CaseType]):
    similarities: dict[CaseName, float]
    ranking: list[CaseName]
    casebase: Casebase[CaseType]
