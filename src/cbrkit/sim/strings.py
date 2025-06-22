import csv
import fnmatch
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from ..helpers import optional_dependencies
from ..typing import FilePath, JsonDict, SimFunc
from .generic import static_table

__all__ = [
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "ngram",
    "regex",
    "glob",
    "table",
]


@dataclass(slots=True)
class table(static_table[str]):
    """Allows to import a similarity values from a table with optional case-insensitive matching.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for batches not in the table
        case_sensitive: If True, the comparison is case-sensitive

    Examples:
        >>> sim = table(
        ...     [("a", "b", 0.5), ("b", "c", 0.7)],
        ...     symmetric=True,
        ...     default=0.0
        ... )
        >>> sim("b", "a")
        0.5
        >>> sim("a", "c")
        0.0
    """

    case_sensitive: bool

    def __init__(
        self,
        entries: Sequence[tuple[str, str, float]]
        | Mapping[tuple[str, str], float]
        | FilePath,
        default: float = 0.0,
        symmetric: bool = True,
        case_sensitive: bool = True,
    ):
        self.case_sensitive = case_sensitive

        # Parse entries from file if needed
        if isinstance(entries, str | Path):
            if isinstance(entries, str):
                entries = Path(entries)

            if entries.suffix != ".csv":
                raise NotImplementedError()

            with entries.open() as f:
                reader = csv.reader(f)
                if case_sensitive:
                    parsed_entries = [(x, y, float(z)) for x, y, z in reader]
                else:
                    parsed_entries = [
                        (x.lower(), y.lower(), float(z)) for x, y, z in reader
                    ]
        else:
            if case_sensitive:
                parsed_entries = entries
            else:
                if isinstance(entries, Mapping):
                    parsed_entries = {
                        (k[0].lower(), k[1].lower()): v for k, v in entries.items()
                    }
                else:
                    parsed_entries = [(x.lower(), y.lower(), z) for x, y, z in entries]

        # Call parent constructor
        super(table, self).__init__(
            parsed_entries, default=default, symmetric=symmetric
        )

    @property
    @override
    def metadata(self) -> JsonDict:
        meta = super(table, self).metadata
        meta["case_sensitive"] = self.case_sensitive
        return meta

    @override
    def __call__(self, x: str, y: str) -> float:
        if not self.case_sensitive:
            x = x.lower()
            y = y.lower()
        return super(table, self).__call__(x, y)


@dataclass(slots=True, frozen=True)
class regex(SimFunc[str, float]):
    """Compares a case x to a query y, written as a regular expression. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Args:
        case_sensitive: If True, the comparison is case-sensitive
    Examples:
        >>> sim = regex()
        >>> sim("Test1", "T.st[0-9]")
        1.0
        >>> sim("Test2", "T.st[3-6]")
        0.0
    """

    case_sensitive: bool = True

    @override
    def __call__(self, x: str, y: str) -> float:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        regex_pattern = re.compile(y, flags)
        return 1.0 if regex_pattern.match(x) else 0.0


@dataclass(slots=True, frozen=True)
class glob(SimFunc[str, float]):
    """Compares a case x to a query y, written as a glob pattern, which can contain wildcards. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Args:
        case_sensitive: If True, the comparison is case-sensitive
    Examples:
        >>> sim = glob()
        >>> sim("Test1", "Test?")
        1.0
        >>> sim("Test2", "Test[3-9]")
        0.0
    """

    case_sensitive: bool = True

    @override
    def __call__(self, x: str, y: str) -> float:
        comparison_func = (
            fnmatch.fnmatchcase if self.case_sensitive else fnmatch.fnmatch
        )
        return 1.0 if comparison_func(x, y) else 0.0


with optional_dependencies():
    import Levenshtein as levenshtein_lib

    @dataclass(slots=True, frozen=True)
    class levenshtein(SimFunc[str, float]):
        """Similarity function that calculates a normalized indel similarity between two strings based on [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance).

        Args:
            score_cutoff: If the similarity is less than this value, the function will return 0.0.
        Examples:
            >>> sim = levenshtein()
            >>> sim("kitten", "sitting")
            0.6153846153846154
            >>> sim = levenshtein(score_cutoff=0.8)
            >>> sim("kitten", "sitting")
            0.0
        """

        score_cutoff: float | None = None
        case_sensitive: bool = True

        @override
        def __call__(self, x: str, y: str) -> float:
            if not self.case_sensitive:
                x, y = x.lower(), y.lower()

            return levenshtein_lib.ratio(x, y, score_cutoff=self.score_cutoff)

    @dataclass(slots=True, frozen=True)
    class jaro(SimFunc[str, float]):
        """Jaro similarity function to compute similarity between two strings.

        Args:
            score_cutoff: If the similarity is less than this value, the function will return 0.0.
            case_sensitive: If True, the comparison is case-sensitive
        Examples:
            >>> sim = jaro()
            >>> sim("kitten", "sitting")
            0.746031746031746
            >>> sim = jaro(score_cutoff=0.8)
            >>> sim("kitten", "sitting")
            0.0
        """

        score_cutoff: float | None = None
        case_sensitive: bool = True

        @override
        def __call__(self, x: str, y: str) -> float:
            if not self.case_sensitive:
                x, y = x.lower(), y.lower()

            return levenshtein_lib.jaro(x, y, score_cutoff=self.score_cutoff)

    @dataclass(slots=True, frozen=True)
    class jaro_winkler(SimFunc[str, float]):
        """Jaro-Winkler similarity function to compute similarity between two strings.

        Args:
            score_cutoff: If the similarity is less than this value, the function will return 0.0.
            prefix_weight: Weight used for the common prefix of the two strings. Has to be between 0 and 0.25. Default is 0.1.
            case_sensitive: If True, the comparison is case-sensitive
        Examples:
            >>> sim = jaro_winkler()
            >>> sim("kitten", "sitting")
            0.746031746031746
            >>> sim = jaro_winkler(score_cutoff=0.8)
            >>> sim("kitten", "sitting")
            0.0
        """

        score_cutoff: float | None = None
        prefix_weight: float = 0.1
        case_sensitive: bool = True

        @override
        def __call__(self, x: str, y: str) -> float:
            if not self.case_sensitive:
                x, y = x.lower(), y.lower()

            return levenshtein_lib.jaro_winkler(
                x, y, score_cutoff=self.score_cutoff, prefix_weight=self.prefix_weight
            )


with optional_dependencies():
    from nltk.util import ngrams as nltk_ngrams

    @dataclass(slots=True, frozen=True)
    class ngram(SimFunc[str, float]):
        """N-gram similarity function to compute [similarity](https://procake.pages.gitlab.rlp.net/procake-wiki/sim/strings/#n-gram) between two strings.

        Args:
            n: Length of the n-gram
            case_sensitive: If True, the comparison is case-sensitive
            tokenizer: Tokenizer function to split the input strings into tokens. If None, the input strings are split into characters.
        Examples:
            >>> sim = ngram(3, case_sensitive=False)
            >>> sim("kitten", "sitting")
            0.125

        """

        n: int
        case_sensitive: bool = True
        tokenizer: Callable[[str], Sequence[str]] | None = None

        @override
        def __call__(self, x: str, y: str) -> float:
            if not self.case_sensitive:
                x = x.lower()
                y = y.lower()

            x_items = self.tokenizer(x) if self.tokenizer is not None else list(x)
            y_items = self.tokenizer(y) if self.tokenizer is not None else list(y)

            x_ngrams = set(nltk_ngrams(x_items, self.n))
            y_ngrams = set(nltk_ngrams(y_items, self.n))

            return len(x_ngrams.intersection(y_ngrams)) / len(x_ngrams.union(y_ngrams))
