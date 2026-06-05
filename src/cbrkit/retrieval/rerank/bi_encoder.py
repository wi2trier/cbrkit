"""Bi-encoder re-ranking via sentence-transformers (embedding similarity)."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import override

from sentence_transformers import SentenceTransformer, util

from ...helpers import run_threaded
from ..wrappers import synced
from ._common import RerankFunc


@dataclass(slots=True)
class bi_encoder[K](RerankFunc[K]):
    """Bi-encoder re-ranking with [sentence-transformers](https://www.sbert.net/).

    Re-scores candidates by cosine similarity between independently encoded
    query and document embeddings.  Cheaper but less accurate than a
    :class:`cross_encoder`, which jointly encodes each ``(query, document)``
    pair; for that reason a cross-encoder is the better default reranker.

    Args:
        model: Name of a [sentence-transformer model](https://www.sbert.net/docs/pretrained_models.html)
            or a ``SentenceTransformer`` instance.
        batch_size: Number of texts encoded per forward pass.
        device: Torch device passed to ``SentenceTransformer`` when *model* is a name.
    """

    model: SentenceTransformer | str
    batch_size: int = 32
    device: str | None = None
    _model: SentenceTransformer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._model = (
            SentenceTransformer(self.model, device=self.device)
            if isinstance(self.model, str)
            else self.model
        )

    @override
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        return await run_threaded(self._score, query, documents)

    def _score(self, query: str, documents: list[str]) -> list[tuple[int, float]]:
        embeddings = util.normalize_embeddings(
            self._model.encode(
                [query, *documents], convert_to_tensor=True, batch_size=self.batch_size
            )
        )
        scores = util.cos_sim(embeddings[:1], embeddings[1:])[0]
        return [(index, float(score)) for index, score in enumerate(scores)]


def sentence_transformers[K](
    model: SentenceTransformer | str,
    batch_size: int = 32,
    device: str | None = None,
) -> synced[K, str, float]:
    """Deprecated sync alias for the `bi_encoder` reranker.

    Wraps `bi_encoder` in `synced` so it keeps the pre-async behaviour of
    running inside a synchronous pipeline (`apply_query` / `apply_queries`).
    Prefer `bi_encoder` directly in an async pipeline.
    """
    return synced(bi_encoder[K](model, batch_size, device))


__all__ = ["bi_encoder", "sentence_transformers"]
