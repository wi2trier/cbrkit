"""
**Please note:** Taxonomy-based similarities are available in the `cbrkit.sim.strings.taxonomy` module.
"""

import csv
import fnmatch
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, override

from ...typing import (
    FilePath,
    JsonDict,
    SimPairFunc,
    SimSeq,
    SimSeqFunc,
    SupportsMetadata,
)
from ..generic import static_table
from . import taxonomy

__all__ = [
    "table",
    "taxonomy",
    "ngram",
    "regex",
    "glob",
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


try:
    from spacy import load as spacy_load
    from spacy.language import Language

    @dataclass(slots=True)
    class spacy(SimSeqFunc[str, float], SupportsMetadata):
        """Semantic similarity using [spaCy](https://spacy.io/)

        Args:
            model: Either the name of a [spaCy model](https://spacy.io/usage/models)
                or a `spacy.Language` model instance.
        """

        model: Language

        def __init__(self, model: str | Language):
            if isinstance(model, str):
                self.model = spacy_load(model)
            else:
                self.model = model

        @property
        @override
        def metadata(self) -> JsonDict:
            return {"model": self.model.meta}

        @override
        def __call__(self, pairs: Sequence[tuple[str, str]]) -> SimSeq[float]:
            texts = _unique_items(pairs)

            with self.model.select_pipes(enable=[]):
                _docs = self.model.pipe(texts)

            docs = dict(zip(texts, _docs, strict=True))

            return [docs[x].similarity(docs[y]) for x, y in pairs]

    __all__ += ["spacy"]

except ImportError:
    pass


try:
    from sentence_transformers import SentenceTransformer

    @dataclass(slots=True)
    class sentence_transformers(SimSeqFunc[str, float], SupportsMetadata):
        """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

        Args:
            model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
                or a `SentenceTransformer` model instance.
        """

        model: SentenceTransformer
        _metadata: JsonDict = field(default_factory=dict, init=False)

        def __init__(self, model: str | SentenceTransformer):
            if isinstance(model, str):
                self.model = SentenceTransformer(model)
                self._metadata["model"] = model
            else:
                self.model = model
                self._metadata["model"] = "custom"

        @property
        @override
        def metadata(self) -> JsonDict:
            return self._metadata

        @override
        def __call__(self, pairs: Sequence[tuple[str, str]]) -> SimSeq[float]:
            texts = _unique_items(pairs)
            encoded_texts = self.model.encode(texts, convert_to_numpy=True)
            vecs = dict(zip(texts, encoded_texts, strict=True))

            return [_cosine(vecs[x], vecs[y]) for x, y in pairs]

    __all__ += ["sentence_transformers"]

except ImportError:
    pass


try:
    import numpy as np
    from openai import OpenAI

    @dataclass(slots=True, frozen=True)
    class openai(SimSeqFunc[str, float], SupportsMetadata):
        """Semantic similarity using OpenAI's embedding models

        Args:
            model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
        """

        model: str
        client: OpenAI = field(default_factory=OpenAI)

        @property
        @override
        def metadata(self) -> JsonDict:
            return {"model": self.model}

        @override
        def __call__(self, pairs: Sequence[tuple[str, str]]) -> SimSeq:
            texts = _unique_items(pairs)
            res = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
            )
            _vecs = [np.array(x.embedding) for x in res.data]
            vecs = dict(zip(texts, _vecs, strict=True))

            return [_cosine(vecs[x], vecs[y]) for x, y in pairs]

    __all__ += ["openai"]

except ImportError:
    pass


try:
    import numpy as np
    from ollama import Client, Options

    @dataclass(slots=True, frozen=True)
    class ollama(SimSeqFunc[str, float], SupportsMetadata):
        """Semantic similarity using Ollama's embedding models

        Args:
            model: Name of the [embedding model](https://ollama.com/blog/embedding-models).
        """

        model: str
        truncate: bool = True
        options: Options | None = None
        keep_alive: float | str | None = None
        client: Client = field(default_factory=Client)

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "model": self.model,
                "truncate": self.truncate,
                "keep_alive": self.keep_alive,
                "options": str(self.options),
            }

        @override
        def __call__(self, pairs: Sequence[tuple[str, str]]) -> SimSeq:
            texts = _unique_items(pairs)
            res = self.client.embed(
                self.model,
                texts,
                truncate=self.truncate,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            _vecs = [np.array(x) for x in res["embeddings"]]
            vecs = dict(zip(texts, _vecs, strict=True))

            return [_cosine(vecs[x], vecs[y]) for x, y in pairs]

    __all__ += ["ollama"]

except ImportError:
    pass


try:
    import numpy as np
    from cohere import Client
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere(SimSeqFunc[str, float], SupportsMetadata):
        """Semantic similarity using Cohere's embedding models

        Args:
            model: Name of the [embedding model](https://docs.cohere.com/reference/embed).
        """

        model: str
        client: Client = field(default_factory=Client)
        truncate: Literal["NONE", "START", "END"] | None = None
        request_options: RequestOptions | None = None

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "model": self.model,
                "truncate": self.truncate,
                "request_options": str(self.request_options),
            }

        @override
        def __call__(self, pairs: Sequence[tuple[str, str]]) -> SimSeq:
            case_texts = list({x for x, _ in pairs})
            query_texts = list({y for _, y in pairs})

            case_raw_vecs = self.client.v2.embed(
                model=self.model,
                texts=case_texts,
                input_type="search_document",
                embedding_types="float",
                truncate=self.truncate,
                request_options=self.request_options,
            ).embeddings.float_
            query_raw_vecs = self.client.v2.embed(
                model=self.model,
                texts=query_texts,
                input_type="search_document",
                embedding_types="float",
                truncate=self.truncate,
                request_options=self.request_options,
            ).embeddings.float_

            assert case_raw_vecs is not None and query_raw_vecs is not None

            case_np_vecs = [np.array(x) for x in case_raw_vecs]
            query_np_vecs = [np.array(x) for x in query_raw_vecs]

            case_vecs = dict(zip(case_texts, case_np_vecs, strict=True))
            query_vecs = dict(zip(query_texts, query_np_vecs, strict=True))

            return [_cosine(case_vecs[x], query_vecs[y]) for x, y in pairs]

    __all__ += ["cohere"]

except ImportError:
    pass


try:
    import Levenshtein as pyLevenshtein

    @dataclass(slots=True, frozen=True)
    class levenshtein(SimPairFunc[str, float], SupportsMetadata):
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

            return pyLevenshtein.ratio(x, y, score_cutoff=self.score_cutoff)

    @dataclass(slots=True, frozen=True)
    class jaro(SimPairFunc[str, float], SupportsMetadata):
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
            return pyLevenshtein.jaro(x, y, score_cutoff=self.score_cutoff)

    @dataclass(slots=True, frozen=True)
    class jaro_winkler(SimPairFunc[str, float], SupportsMetadata):
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
            return pyLevenshtein.jaro_winkler(
                x, y, score_cutoff=self.score_cutoff, prefix_weight=self.prefix_weight
            )

    __all__ += ["levenshtein", "jaro", "jaro_winkler"]

except ImportError:
    pass


try:
    from nltk.util import ngrams as nltk_ngrams

    @dataclass(slots=True, frozen=True)
    class ngram(SimPairFunc[str, float], SupportsMetadata):
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

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "n": self.n,
                "case_sensitive": self.case_sensitive,
                "tokenizer": self.tokenizer is not None,
            }

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

    __all__ += ["ngram"]

except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class regex(SimPairFunc[str, float], SupportsMetadata):
    """Compares a case x to a query y, written as a regular expression. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Examples:
        >>> sim = regex()
        >>> sim("Test1", "T.st[0-9]")
        1.0
        >>> sim("Test2", "T.st[3-6]")
        0.0
    """

    @override
    def __call__(self, x: str, y: str) -> float:
        regex = re.compile(y)
        return 1.0 if regex.match(x) else 0.0


@dataclass(slots=True, frozen=True)
class glob(SimPairFunc[str, float], SupportsMetadata):
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

    case_sensitive: bool = False

    @override
    def __call__(self, x: str, y: str) -> float:
        comparison_func = (
            fnmatch.fnmatchcase if self.case_sensitive else fnmatch.fnmatch
        )
        return 1.0 if comparison_func(x, y) else 0.0


def table(
    entries: Sequence[tuple[str, str, float]]
    | Mapping[tuple[str, str], float]
    | FilePath,
    symmetric: bool = True,
    default: float = 0.0,
) -> SimPairFunc[str, float]:
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for pairs not in the table

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
    if isinstance(entries, str | Path):
        if isinstance(entries, str):
            entries = Path(entries)

        if entries.suffix != ".csv":
            raise NotImplementedError()

        with entries.open() as f:
            reader = csv.reader(f)
            parsed_entries = [(x, y, float(z)) for x, y, z in reader]

    else:
        parsed_entries = entries

    return static_table(parsed_entries, symmetric=symmetric, default=default)
