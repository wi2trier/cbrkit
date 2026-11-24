"""BM25 sparse embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast, override

import bm25s
import Stemmer  # ty: ignore[import-not-found]
from bm25s.tokenization import Tokenized

from ....typing import EmbeddingFunc, SparseVector


def tokenize(
    texts: Sequence[str], language: str, stopwords: Sequence[str] | None
) -> Tokenized:
    """Tokenize texts with the stemmer and the stopwords of a language.

    Args:
        texts: The texts.
        language: Language of the stemmer, and of the stopwords if they are None.
        stopwords: The stopwords, those of the language if None.
    """
    return cast(
        Tokenized,
        bm25s.tokenize(
            list(texts),
            stemmer=Stemmer.Stemmer(language),
            stopwords=language if stopwords is None else list(stopwords),
        ),
    )


def index(
    texts: Sequence[str], language: str, stopwords: Sequence[str] | None
) -> bm25s.BM25:
    """Build the BM25 index of a corpus.

    Args:
        texts: The corpus.
        language: Language of the stemmer, and of the stopwords if they are None.
        stopwords: The stopwords, those of the language if None.
    """
    model = bm25s.BM25()
    model.index(tokenize(texts, language, stopwords))

    return model


@dataclass(slots=True, frozen=True, kw_only=True)
class BM25Encoder(EmbeddingFunc[str, SparseVector]):
    """BM25-based sparse embeddings over the vocabulary of a corpus using
    `bm25s <https://github.com/xhluca/bm25s>`_.

    Produces sparse vectors where each dimension corresponds to a
    vocabulary token of the corpus and the value represents the term frequency.
    The vocabulary is derived from the corpus once, when the function is created.

    Args:
        corpus: The texts whose vocabulary spans the dimensions.
        language: Language for stemming and stopwords.
        stopwords: The stopwords, those of the language if None.

    Examples:
        >>> import asyncio
        >>> embed = BM25Encoder(corpus=["red car", "blue bike"])
        >>> [len(vector) for vector in asyncio.run(embed(["red bike", "green"]))]
        [2, 0]
    """

    corpus: Sequence[str]
    language: str = "english"
    stopwords: Sequence[str] | None = None
    _vocabulary: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_vocabulary",
            index(self.corpus, self.language, self.stopwords).vocab_dict
            if self.corpus
            else {},
        )

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[SparseVector]:
        if not values:
            return []

        tokenized = tokenize(values, self.language, self.stopwords)
        tokens = {token_id: token for token, token_id in tokenized.vocab.items()}
        result: list[SparseVector] = []

        for token_ids in tokenized.ids:
            vector: SparseVector = {}

            for token_id in token_ids:
                token = tokens.get(int(token_id))

                if token is not None and token in self._vocabulary:
                    dimension = self._vocabulary[token]
                    vector[dimension] = vector.get(dimension, 0.0) + 1.0

            result.append(vector)

        return result


__all__ = ["BM25Encoder", "index", "tokenize"]
