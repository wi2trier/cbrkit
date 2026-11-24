import fnmatch
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, ClassVar, Literal, override

import numpy as np
from pydantic import Field
from rapidfuzz import distance, process

from .._util import fold
from ..schema import PythonOnly, Unit, serializable
from ..typing import ColumnSimFunc, NumpyArray, SimFunc

__all__ = [
    "Glob",
    "Indel",
    "Jaro",
    "JaroWinkler",
    "Levenshtein",
    "Ngram",
    "Regex",
    "WordCoverage",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Regex(SimFunc[str, float]):
    """Compares a case x to a query y, written as a regular expression. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Args:
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = Regex()
        >>> sim("Test1", "T.st[0-9]")
        1.0
        >>> sim("Test2", "T.st[3-6]")
        0.0
    """

    case_sensitive: bool = True
    kind: Literal["regex"] = field(default="regex", repr=False)

    @override
    def __call__(self, case: str, query: str) -> float:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        regex_pattern = re.compile(query, flags)
        return 1.0 if regex_pattern.match(case) else 0.0


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Glob(SimFunc[str, float]):
    """Compares a case x to a query y, written as a glob pattern, which can contain wildcards. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Args:
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = Glob()
        >>> sim("Test1", "Test?")
        1.0
        >>> sim("Test2", "Test[3-9]")
        0.0
    """

    case_sensitive: bool = True
    kind: Literal["glob"] = field(default="glob", repr=False)

    @override
    def __call__(self, case: str, query: str) -> float:
        comparison_func = (
            fnmatch.fnmatchcase if self.case_sensitive else fnmatch.fnmatch
        )
        return 1.0 if comparison_func(case, query) else 0.0


class _RapidFuzz(ColumnSimFunc[str]):
    """Base of the measures that compare strings with a scorer of rapidfuzz."""

    __slots__ = ()

    _scorer: ClassVar[staticmethod[..., float]]
    score_cutoff: Unit | None
    case_sensitive: bool

    def _kwargs(self) -> dict[str, float]:
        """Further keyword arguments of the scorer."""
        return {}

    @override
    def __call__(self, case: str, query: str) -> float:
        case, query = fold(case, self.case_sensitive), fold(query, self.case_sensitive)

        return self._scorer(
            case, query, score_cutoff=self.score_cutoff, **self._kwargs()
        )

    @override
    def columns(self, cases: Sequence[str], queries: Sequence[str]) -> NumpyArray:
        return process.cpdist(
            [fold(case, self.case_sensitive) for case in cases],
            [fold(query, self.case_sensitive) for query in queries],
            scorer=self._scorer,
            score_cutoff=self.score_cutoff,
            dtype=np.float64,
            scorer_kwargs=self._kwargs(),
        )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Levenshtein(_RapidFuzz):
    """Normalized [Levenshtein](https://en.wikipedia.org/wiki/Levenshtein_distance) similarity, i.e., one minus the edit distance divided by the length of the longer string.

    Args:
        score_cutoff: If the similarity is less than this value, the function will return 0.0.
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = Levenshtein()
        >>> sim("kitten", "sitting")
        0.5714285714285714
        >>> sim = Levenshtein(score_cutoff=0.8)
        >>> sim("kitten", "sitting")
        0.0
    """

    score_cutoff: Unit | None = None
    case_sensitive: bool = True
    kind: Literal["levenshtein"] = field(default="levenshtein", repr=False)

    _scorer = staticmethod(distance.Levenshtein.normalized_similarity)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Indel(_RapidFuzz):
    """Normalized [Indel](https://en.wikipedia.org/wiki/Edit_distance) similarity, i.e., an edit distance that only allows insertions and deletions.

    Args:
        score_cutoff: If the similarity is less than this value, the function will return 0.0.
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = Indel()
        >>> sim("kitten", "sitting")
        0.6153846153846154
        >>> sim = Indel(score_cutoff=0.8)
        >>> sim("kitten", "sitting")
        0.0
    """

    score_cutoff: Unit | None = None
    case_sensitive: bool = True
    kind: Literal["indel"] = field(default="indel", repr=False)

    _scorer = staticmethod(distance.Indel.normalized_similarity)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Jaro(_RapidFuzz):
    """Jaro similarity function to compute similarity between two strings.

    Args:
        score_cutoff: If the similarity is less than this value, the function will return 0.0.
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = Jaro()
        >>> sim("kitten", "sitting")
        0.746031746031746
        >>> sim = Jaro(score_cutoff=0.8)
        >>> sim("kitten", "sitting")
        0.0
    """

    score_cutoff: Unit | None = None
    case_sensitive: bool = True
    kind: Literal["jaro"] = field(default="jaro", repr=False)

    _scorer = staticmethod(distance.Jaro.similarity)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class JaroWinkler(_RapidFuzz):
    """Jaro-Winkler similarity function to compute similarity between two strings.

    Args:
        score_cutoff: If the similarity is less than this value, the function will return 0.0.
        prefix_weight: Weight used for the common prefix of the two strings. Has to be between 0 and 0.25. Default is 0.1.
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = JaroWinkler()
        >>> sim("kitten", "sitting")
        0.746031746031746
        >>> sim = JaroWinkler(score_cutoff=0.8)
        >>> sim("kitten", "sitting")
        0.0
    """

    score_cutoff: Unit | None = None
    prefix_weight: Annotated[float, Field(ge=0, le=0.25)] = 0.1
    case_sensitive: bool = True
    kind: Literal["jaro_winkler"] = field(default="jaro_winkler", repr=False)

    _scorer = staticmethod(distance.JaroWinkler.similarity)

    @override
    def _kwargs(self) -> dict[str, float]:
        return {"prefix_weight": self.prefix_weight}


def _ngrams(items: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(items[i : i + n]) for i in range(len(items) - n + 1)}


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Ngram(SimFunc[str, float]):
    """N-gram similarity function to compute [similarity](https://procake.pages.gitlab.rlp.net/procake-wiki/sim/strings/#n-gram) between two strings.

    Args:
        n: Length of the n-gram
        case_sensitive: If True, the comparison is case-sensitive
        tokenizer: Tokenizer function to split the input strings into tokens. If None, the input strings are split into characters.

    Examples:
        >>> sim = Ngram(n=3, case_sensitive=False)
        >>> sim("kitten", "sitting")
        0.125

    """

    n: Annotated[int, Field(ge=1)]
    case_sensitive: bool = True
    tokenizer: PythonOnly[Callable[[str], Sequence[str]] | None] = None
    kind: Literal["ngram"] = field(default="ngram", repr=False)

    @override
    def __call__(self, case: str, query: str) -> float:
        case, query = fold(case, self.case_sensitive), fold(query, self.case_sensitive)
        x_items = self.tokenizer(case) if self.tokenizer is not None else list(case)
        y_items = self.tokenizer(query) if self.tokenizer is not None else list(query)

        x_ngrams = _ngrams(x_items, self.n)
        y_ngrams = _ngrams(y_items, self.n)
        union = x_ngrams | y_ngrams

        # both inputs are shorter than n
        if not union:
            return 1.0 if x_items == y_items else 0.0

        return len(x_ngrams & y_ngrams) / len(union)


_WORDS = re.compile(r"\w+")


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class WordCoverage(SimFunc[str, float]):
    """Share of the distinct words of the query that the case contains, such as for searching texts by keywords.

    It is asymmetric: a long case that contains every word of a short query is fully similar to it.

    Args:
        case_sensitive: If True, the comparison is case-sensitive.

    Examples:
        >>> sim = WordCoverage(case_sensitive=False)
        >>> sim("The cat sat on the mat", "cat dog")
        0.5
        >>> sim("anything", "")
        0.0
    """

    case_sensitive: bool = True
    kind: Literal["word_coverage"] = field(default="word_coverage", repr=False)

    @override
    def __call__(self, case: str, query: str) -> float:
        case, query = fold(case, self.case_sensitive), fold(query, self.case_sensitive)
        words = set(_WORDS.findall(query))

        if not words:
            return 0.0

        return len(words & set(_WORDS.findall(case))) / len(words)
