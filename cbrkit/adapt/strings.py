import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from cbrkit.helpers import get_metadata
from cbrkit.typing import AdaptPairFunc, JsonDict, SupportsMetadata

__all__ = [
    "regex",
]


@dataclass(slots=True)
class regex(AdaptPairFunc[str], SupportsMetadata):
    case_pattern: re.Pattern[str]
    query_pattern: re.Pattern[str]
    replacement: Callable[[re.Match[str], re.Match[str]], str]
    count: int
    pos: int
    endpos: int
    _metadata: JsonDict

    def __init__(
        self,
        case_pattern: str | re.Pattern[str],
        query_pattern: str | re.Pattern[str],
        replacement: str | Callable[[re.Match[str], re.Match[str]], str],
        count: int = 0,
        pos: int = 0,
        endpos: int = sys.maxsize,
    ):
        if isinstance(query_pattern, str):
            self.query_pattern = re.compile(query_pattern)
        else:
            self.query_pattern = query_pattern

        if isinstance(case_pattern, str):
            self.case_pattern = re.compile(case_pattern)
        else:
            self.case_pattern = case_pattern

        if isinstance(replacement, str):
            self.replacement = lambda case_match, query_match: replacement
        else:
            self.replacement = replacement

        self.count = count
        self.pos = pos
        self.endpos = endpos

        self._metadata = {
            "query_pattern": str(query_pattern),
            "case_pattern": str(case_pattern),
            "replacement": replacement
            if isinstance(replacement, str)
            else get_metadata(replacement),
            "count": count,
            "pos": pos,
            "endpos": endpos,
        }

    @property
    @override
    def metadata(self) -> JsonDict:
        return self._metadata

    @override
    def __call__(self, case: str, query: str) -> str:
        query_match = self.query_pattern.search(
            query,
            self.pos,
            self.endpos,
        )

        if query_match is not None:
            return self.case_pattern.sub(
                lambda case_match: self.replacement(case_match, query_match),
                case,
                self.count,
            )

        return case
