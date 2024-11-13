import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

from cbrkit.typing import AdaptPairFunc, JsonDict, SupportsMetadata


@dataclass(slots=True)
class regex(AdaptPairFunc[str], SupportsMetadata):
    query_pattern: re.Pattern[str]
    case_pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]
    _metadata: JsonDict

    def __init__(self, query_pattern: str, case_pattern: str, replacement: str):
        self.query_pattern = re.compile(query_pattern)
        self.case_pattern = re.compile(case_pattern)
        self.replacement = replacement

        self._metadata = {
            "query_pattern": query_pattern,
            "case_pattern": case_pattern,
            "replacement": replacement,
        }

    @property
    @override
    def metadata(self) -> JsonDict:
        return self._metadata

    @override
    def __call__(self, case: str, query: str) -> str:
        if self.query_pattern.search(query) is not None:
            return self.case_pattern.sub(self.replacement, case)

        return case
