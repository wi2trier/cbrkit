"""
**Please note:** Taxonomy-based similarities are available in the `cbrkit.sim.strings.taxonomy` module.
"""

import csv
import fnmatch
import itertools
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from cbrkit.sim.generic import table as generic_table
from cbrkit.sim.strings import taxonomy
from cbrkit.typing import (
    FilePath,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
)

if TYPE_CHECKING:
    try:
        from openai import OpenAI
        from sentence_transformers import SentenceTransformer
        from spacy.language import Language as SpacyLanguage
    except ImportError:
        pass

__all__ = [
    "spacy",
    "sentence_transformers",
    "openai",
    "levenshtein",
    "jaro",
    "jaro_winkler",
    "table",
    "taxonomy",
]


def _cosine(u, v) -> float:
    """Cosine similarity between two vectors

    Args:
        u: First vector
        v: Second vector
    """
    import numpy as np
    import scipy.spatial.distance as scipy_dist

    if np.any(u) and np.any(v):
        return 1 - cast(float, scipy_dist.cosine(u, v))

    return 0.0


def _unique_items(pairs: Sequence[tuple[str, str]]) -> list[str]:
    return [*{*itertools.chain.from_iterable(pairs)}]


def spacy(model: str | SpacyLanguage) -> SimSeqFunc[str, float]:
    """Semantic similarity using [spaCy](https://spacy.io/)

    Args:
        model: Either the name of a [spaCy model](https://spacy.io/usage/models)
            or a `spacy.Language` model instance.
    """
    from spacy import load as spacy_load

    if isinstance(model, str):
        model = spacy_load(model)

    def wrapped_func(pairs: Sequence[tuple[str, str]]) -> SimSeq:
        texts = _unique_items(pairs)

        with model.select_pipes(enable=[]):
            _docs = model.pipe(texts)

        docs = dict(zip(texts, _docs, strict=True))

        return [docs[x].similarity(docs[y]) for x, y in pairs]

    return wrapped_func


def sentence_transformers(model: str | SentenceTransformer) -> SimSeqFunc[str, float]:
    """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

    Args:
        model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
            or a `SentenceTransformer` model instance.
    """
    from sentence_transformers import SentenceTransformer

    if isinstance(model, str):
        model = SentenceTransformer(model)

    def wrapped_func(pairs: Sequence[tuple[str, str]]) -> SimSeq:
        texts = _unique_items(pairs)
        encoded_texts = model.encode(texts, convert_to_numpy=True)
        vecs = dict(zip(texts, encoded_texts, strict=True))

        return [_cosine(vecs[x], vecs[y]) for x, y in pairs]

    return wrapped_func


def openai(model: str, client: None | OpenAI) -> SimSeqFunc[str, float]:
    """Semantic similarity using OpenAI's embedding models

    Args:
        model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
    """
    import numpy as np
    from openai import OpenAI

    if client is None:
        client = OpenAI()

    def wrapped_func(pairs: Sequence[tuple[str, str]]) -> SimSeq:
        texts = _unique_items(pairs)
        res = client.embeddings.create(input=texts, model=model)
        _vecs = [np.array(x.embedding) for x in res.data]
        vecs = dict(zip(texts, _vecs, strict=True))

        return [_cosine(vecs[x], vecs[y]) for x, y in pairs]

    return wrapped_func


def levenshtein(
    score_cutoff: float | None = None, case_sensitive: bool = True
) -> SimPairFunc[str, float]:
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
    import Levenshtein

    def wrapped_func(x: str, y: str) -> float:
        if not case_sensitive:
            x, y = x.lower(), y.lower()

        return Levenshtein.ratio(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro(score_cutoff: float | None = None) -> SimPairFunc[str, float]:
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
    import Levenshtein

    def wrapped_func(x: str, y: str) -> float:
        return Levenshtein.jaro(x, y, score_cutoff=score_cutoff)

    return wrapped_func


def jaro_winkler(
    score_cutoff: float | None = None, prefix_weight: float = 0.1
) -> SimPairFunc[str, float]:
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
    import Levenshtein

    def wrapped_func(x: str, y: str) -> float:
        return Levenshtein.jaro_winkler(
            x, y, score_cutoff=score_cutoff, prefix_weight=prefix_weight
        )

    return wrapped_func


def ngram(
    n: int,
    case_sensitive: bool = False,
    tokenizer: Callable[[str], Sequence[str]] | None = None,
) -> SimPairFunc[str, float]:
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
    from nltk.util import ngrams

    def wrapped_func(x: str, y: str) -> float:
        if not case_sensitive:
            x = x.lower()
            y = y.lower()

        x_items = tokenizer(x) if tokenizer is not None else list(x)
        y_items = tokenizer(y) if tokenizer is not None else list(y)

        x_ngrams = set(ngrams(x_items, n))
        y_ngrams = set(ngrams(y_items, n))

        return len(x_ngrams.intersection(y_ngrams)) / len(x_ngrams.union(y_ngrams))

    return wrapped_func


def regex() -> SimPairFunc[str, float]:
    """Compares a case x to a query y, written as a regular expression. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Examples:
        >>> sim = regex()
        >>> sim("Test1", "T.st[0-9]")
        1.0
        >>> sim("Test2", "T.st[3-6]")
        0.0
    """

    def wrapped_func(x: str, y: str) -> float:
        regex = re.compile(y)
        return 1.0 if regex.match(x) else 0.0

    return wrapped_func


def glob(case_sensitive: bool = False) -> SimPairFunc[str, float]:
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

    comparison_func = fnmatch.fnmatchcase if case_sensitive else fnmatch.fnmatch

    def wrapped_func(x: str, y: str) -> float:
        return 1.0 if comparison_func(x, y) else 0.0

    return wrapped_func


def table(
    entries: Sequence[tuple[str, str, float]] | FilePath,
    symmetric: bool = True,
    default: float = 0.0,
) -> SimPairFunc[str, float]:
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table

    Examples:
        >>> sim = table([("a", "b", 0.5), ("b", "c", 0.7)], symmetric=True, default=0.0)
        >>> sim("b", "a")
        0.5
        >>> sim("a", "c")
        0.0
    """
    if isinstance(entries, FilePath):
        if isinstance(entries, str):
            entries = Path(entries)

        if entries.suffix != ".csv":
            raise NotImplementedError()

        with entries.open() as f:
            reader = csv.reader(f)
            parsed_entries = [(x, y, float(z)) for x, y, z in reader]

    else:
        parsed_entries = entries

    return generic_table(parsed_entries, symmetric=symmetric, default=default)
