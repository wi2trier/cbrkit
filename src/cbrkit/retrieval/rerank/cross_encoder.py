"""Cross-encoder re-ranking via sentence-transformers."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import override

from sentence_transformers import CrossEncoder

from ...helpers import run_threaded
from ._common import RerankFunc


@dataclass(slots=True)
class cross_encoder[K](RerankFunc[K]):
    """Cross-encoder re-ranking with [sentence-transformers](https://www.sbert.net/).

    A cross-encoder jointly encodes each ``(query, document)`` pair, which is
    markedly more accurate for re-ranking a retrieved candidate set than a
    bi-encoder that embeds query and documents independently.

    Args:
        model: Name of a [cross-encoder model](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
            or a ``CrossEncoder`` instance.
        batch_size: Number of pairs scored per forward pass.
    """

    model: CrossEncoder | str
    batch_size: int = 32
    _model: CrossEncoder = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._model = (
            CrossEncoder(self.model) if isinstance(self.model, str) else self.model
        )

    @override
    async def _rerank(
        self, query: str, documents: list[str]
    ) -> Iterable[tuple[int, float]]:
        scores = await run_threaded(
            self._model.predict,
            [(query, document) for document in documents],
            batch_size=self.batch_size,
        )
        return ((index, float(score)) for index, score in enumerate(scores))


__all__ = ["cross_encoder"]
