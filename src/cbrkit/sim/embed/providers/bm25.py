"""BM25 sparse embedding provider."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import cast, override

import bm25s
import Stemmer  # type: ignore[import-untyped]  # ty: ignore[unresolved-import]
from bm25s.tokenization import Tokenized

from ....indexable import _normalize_patch_keys
from ....typing import BatchConversionFunc, IndexableFunc, SparseVector


@dataclass(slots=True)
class bm25(
    BatchConversionFunc[str, SparseVector],
    IndexableFunc[Collection[str]],
):
    """BM25-based sparse embeddings using
    `bm25s <https://github.com/xhluca/bm25s>`_.

    Produces sparse vectors where each dimension corresponds to a
    vocabulary token and the value represents the term frequency.
    Requires fitting on a corpus via `put_index` before use.

    Args:
        language: Language for stemming and stopwords.
        stopwords: Stopword configuration.  `None` uses the
            language default, a `str` sets the stopwords language
            independently, and a `list[str]` provides custom
            stopwords.
    """

    language: str = "english"
    stopwords: str | list[str] | None = None
    _corpus: list[str] | None = field(default=None, init=False, repr=False)
    _retriever: bm25s.BM25 | None = field(default=None, init=False, repr=False)

    @property
    def _stopwords(self) -> str | list[str]:
        return self.stopwords if self.stopwords is not None else self.language

    @property
    def _stemmer(self) -> Stemmer.Stemmer:
        return Stemmer.Stemmer(self.language)

    def _build_retriever(self, texts: Collection[str]) -> bm25s.BM25:
        tokens = bm25s.tokenize(
            list(texts),
            stemmer=self._stemmer,
            stopwords=self._stopwords,
        )
        retriever = bm25s.BM25()
        retriever.index(tokens)
        return retriever

    @override
    def has_index(self) -> bool:
        """Return whether a BM25 corpus has been indexed."""
        return self._corpus is not None

    @property
    @override
    def index(self) -> Collection[str]:
        """Return the indexed corpus or an empty list if not indexed."""
        if self._corpus is None:
            return []
        return self._corpus

    @override
    def put_index(
        self,
        data: Collection[str],
    ) -> None:
        """Build a new BM25 index from the given corpus."""
        self._corpus = list(data)
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 model from the current corpus."""
        if self._corpus is None:
            self._retriever = None
            return

        if self._corpus:
            self._retriever = self._build_retriever(self._corpus)
        else:
            self._retriever = None

    @override
    def upsert_index(
        self,
        data: Collection[str],
    ) -> None:
        """Add new documents to the existing BM25 index."""
        if self._corpus is None:
            self.put_index(data)
            return

        items = list(data)

        if not items:
            return

        self._corpus.extend(items)
        self._rebuild_index()

    @override
    def delete_index(
        self,
        data: Collection[str],
    ) -> None:
        """Remove the specified documents from the BM25 index."""
        if self._corpus is None:
            return

        if not data:
            return

        remove_set = set(data)
        self._corpus = [t for t in self._corpus if t not in remove_set]
        self._rebuild_index()

    @override
    def patch_index(
        self,
        upsert: Collection[str] | None = None,
        delete: Collection[str] | None = None,
    ) -> None:
        """Apply corpus insertions and deletions."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_set = normalized
        upsert_items = list(upsert) if upsert else []

        if self._corpus is None:
            self.put_index(upsert_items)
            return

        if delete_set:
            self._corpus = [t for t in self._corpus if t not in delete_set]

        self._corpus.extend(upsert_items)
        self._rebuild_index()

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        if not texts:
            return []
        if self._retriever is None:
            raise ValueError("BM25 model must be fitted first. Call put_index().")

        tokenized = cast(
            Tokenized,
            bm25s.tokenize(
                list(texts),
                stemmer=self._stemmer,
                stopwords=self._stopwords,
            ),
        )
        corpus_vocab = self._retriever.vocab_dict
        query_reverse = {v: k for k, v in tokenized.vocab.items()}
        result: list[SparseVector] = []

        for token_ids in tokenized.ids:
            sparse_vec: SparseVector = {}
            for tid in token_ids:
                token_str = query_reverse.get(int(tid))
                if token_str is not None and token_str in corpus_vocab:
                    corpus_id = corpus_vocab[token_str]
                    sparse_vec[corpus_id] = sparse_vec.get(corpus_id, 0.0) + 1.0
            result.append(sparse_vec)

        return result


__all__ = ["bm25"]
