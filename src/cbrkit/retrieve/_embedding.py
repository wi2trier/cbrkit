"""Retrieval by the similarity of embeddings."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .._phase import Ranking
from ..helpers import batchify
from ..sim import Measure
from ..sim.embed import Cosine, embed_pairs
from ..typing import (
    Batches,
    Casebase,
    EmbeddingFunc,
    MatrixSimFunc,
    NumpyArray,
    Results,
)
from ._common import query_groups

__all__ = ["Embedding"]


@dataclass(slots=True, frozen=True, kw_only=True)
class Embedding(Ranking):
    """Retrieves the cases whose embeddings are most similar to that of the query.

    Every case is embedded once per call, concurrently with the queries,
    while `cbrkit.sim.embed.Embedding` in a `Retriever` embeds every pair of case and query.
    To embed every text only once across calls, pass an explicit `cbrkit.sim.embed.EmbeddingCache`.
    It awaits the embedding function, so it is an asynchronous function of the cycle.

    Args:
        embedder: Embeds the cases, and the queries if `query_embedder` is None.
        metric: Similarity of two embeddings.
        query_embedder: Embeds the queries, if they need another model or input type.

    Examples:
        >>> import numpy as np
        >>> from cbrkit.cycle import run_query
        >>> async def embed(texts):
        ...     return [np.array([len(text), 1.0]) for text in texts]
        >>> run_query({"a": "x", "b": "xxxxxx"}, "xxxxx", Embedding(embedder=embed, limit=1)).ranking
        ('b',)
    """

    embedder: EmbeddingFunc[str, NumpyArray]
    metric: Measure[NumpyArray, float] = field(default_factory=Cosine)
    query_embedder: EmbeddingFunc[str, NumpyArray] | None = None

    async def __call__[K](self, batches: Batches[K, str], /) -> Results[K, str]:
        return self.select_scores(
            batches,
            [
                sim_map
                for casebase, queries in query_groups(batches)
                for sim_map in await self._score(queries, casebase)
            ],
        )

    async def _score[K](
        self, queries: Sequence[str], casebase: Casebase[K, str]
    ) -> Sequence[dict[K, float]]:
        case_vecs, query_vecs = await embed_pairs(
            self.embedder, self.query_embedder, list(casebase.values()), queries
        )

        if case_vecs and isinstance(self.metric, MatrixSimFunc):
            keys = list(casebase)
            sims = self.metric.matrix(np.stack(case_vecs), np.stack(query_vecs))

            return [dict(zip(keys, column, strict=True)) for column in sims.T.tolist()]

        metric = batchify(self.metric)

        return [
            dict(
                zip(
                    casebase.keys(),
                    metric([(case_vec, query_vec) for case_vec in case_vecs]),
                    strict=True,
                )
            )
            for query_vec in query_vecs
        ]
