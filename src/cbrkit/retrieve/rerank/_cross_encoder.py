"""Cross-encoder rerank model of sentence-transformers."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import sentence_transformers

from ...helpers import lazy
from ...typing import RerankFunc


def _load(
    model: sentence_transformers.CrossEncoder | str, device: str | None
) -> Callable[..., Any]:
    return (
        sentence_transformers.CrossEncoder(model, device=device)
        if isinstance(model, str)
        else model
    ).predict


@dataclass(slots=True, frozen=True, kw_only=True)
class CrossEncoder(RerankFunc):
    """Cross-encoder rerank model of [sentence-transformers](https://www.sbert.net/), the scores of `Rerank`.

    A cross-encoder jointly encodes each `(query, document)` pair, which is
    markedly more accurate for re-ranking a retrieved candidate set than a
    bi-encoder that embeds query and documents independently.
    The weights of a model given by name are loaded on the first call,
    and the pairs are scored in a worker thread.
    Every document is scored, since the model scores them independently, so the limit of a call is ignored.

    Args:
        model: Name of a [cross-encoder model](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
            or a `CrossEncoder` instance.
        batch_size: Number of pairs scored per forward pass.
        device: Device of a model given by name, such as `"cpu"`, detected if None.
    """

    model: sentence_transformers.CrossEncoder | str
    batch_size: int = 32
    device: str | None = None
    _predict: Callable[..., Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_predict", lazy(partial(_load, self.model, self.device))
        )

    async def __call__(
        self, query: str, documents: Sequence[str], /, limit: int | None = None
    ) -> Mapping[int, float]:
        scores = await asyncio.to_thread(
            self._predict,
            [(query, document) for document in documents],
            batch_size=self.batch_size,
        )

        return {index: float(score) for index, score in enumerate(scores)}


__all__ = ["CrossEncoder"]
