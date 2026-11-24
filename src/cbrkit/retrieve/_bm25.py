"""Retrieval by BM25 text relevance."""

from collections.abc import Sequence
from dataclasses import dataclass

from .._phase import Ranking
from ..sim.embed.providers.bm25 import index, tokenize
from ..typing import Batches, Casebase, Results
from ._common import query_groups

__all__ = ["BM25"]


@dataclass(slots=True, frozen=True, kw_only=True)
class BM25(Ranking):
    """Retrieves the cases that are most relevant to the query by BM25 using `bm25s`.

    The BM25 index of a casebase is built once per call for all queries that share it,
    a persistent text index is the sparse search of a storage of `cbrkit.indexable`.

    Args:
        language: Language for stemming and stopwords.
        stopwords: The stopwords, those of the language if None.
        normalize_scores: Scale the scores of every query to between 0 and 1,
            otherwise return the raw BM25 scores.
    """

    language: str = "english"
    stopwords: Sequence[str] | None = None
    normalize_scores: bool = True

    def __call__[K](self, batches: Batches[K, str], /) -> Results[K, str]:
        return self.select_scores(
            batches,
            [
                sim_map
                for casebase, queries in query_groups(batches)
                for sim_map in self._score(queries, casebase)
            ],
        )

    def _score[K](
        self, queries: Sequence[str], casebase: Casebase[K, str]
    ) -> Sequence[dict[K, float]]:
        if not casebase:
            return [{} for _ in queries]

        keys = list(casebase)
        documents, scores = index(
            list(casebase.values()), self.language, self.stopwords
        ).retrieve(
            tokenize(queries, self.language, self.stopwords),
            sorted=False,
            k=len(keys),
        )

        return [
            {
                keys[int(document)]: float(score)
                for document, score in zip(documents[query], scores[query], strict=True)
            }
            for query in range(len(queries))
        ]
