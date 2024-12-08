from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import override

from ...helpers import optional_dependencies
from ...typing import (
    SimFunc,
)

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
        Examples:
            >>> sim = jaro()
            >>> sim("kitten", "sitting")
            0.746031746031746
            >>> sim = jaro(score_cutoff=0.8)
            >>> sim("kitten", "sitting")
            0.0
        """

        score_cutoff: float | None = None

        @override
        def __call__(self, x: str, y: str) -> float:
            return levenshtein_lib.jaro(x, y, score_cutoff=self.score_cutoff)

    @dataclass(slots=True, frozen=True)
    class jaro_winkler(SimFunc[str, float]):
        """Jaro-Winkler similarity function to compute similarity between two strings.

        Args:
            score_cutoff: If the similarity is less than this value, the function will return 0.0.
            prefix_weight: Weight used for the common prefix of the two strings. Has to be between 0 and 0.25. Default is 0.1.
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

        @override
        def __call__(self, x: str, y: str) -> float:
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
        case_sensitive: bool = False
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
