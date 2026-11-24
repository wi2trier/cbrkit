import re
import sys
from dataclasses import dataclass, field
from typing import Annotated, Literal, override

from pydantic import Field

from ..schema import serializable
from ..typing import AdaptationFunc

__all__ = [
    "RegexAdaptation",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class RegexAdaptation(AdaptationFunc[str]):
    r"""Replace parts of a string using regular expressions.

    The case pattern is only replaced if the query pattern matches the query.

    Args:
        case_pattern: Regular expression pattern to match against the case.
        query_pattern: Regular expression pattern to match against the query.
        replacement: Replacement of the case matches, which may refer to their groups like `re.sub`.
        count: Maximum number of replacements to make, all if None.
        pos: Position in the query to start searching.
        endpos: Position in the query to stop searching.

    Examples:
        >>> func = RegexAdaptation(case_pattern="25", query_pattern="30", replacement="NUMBER")
        >>> func("Alice is 25 years old.", "Peter is 30 years old.")
        'Alice is NUMBER years old.'
        >>> RegexAdaptation(case_pattern=r"(\w+) is", query_pattern="Peter", replacement=r"\1 was")("Alice is 25.", "Peter")
        'Alice was 25.'
    """

    case_pattern: str
    query_pattern: str
    replacement: str
    count: Annotated[int, Field(ge=1)] | None = None
    pos: int = 0
    endpos: int = sys.maxsize
    kind: Literal["regex"] = field(default="regex", repr=False)

    @override
    def __call__(self, case: str, query: str) -> str:
        if re.compile(self.query_pattern).search(query, self.pos, self.endpos) is None:
            return case

        # `re.sub` replaces all for a count of 0.
        return re.sub(self.case_pattern, self.replacement, case, count=self.count or 0)
