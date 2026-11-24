"""sentence-transformers dense embedding provider."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, cast, override

from sentence_transformers import SentenceTransformer

from ....helpers import lazy
from ....typing import EmbeddingFunc, NumpyArray


def _load(model: str | SentenceTransformer, device: str | None) -> Callable[..., Any]:
    return (
        SentenceTransformer(model, device=device) if isinstance(model, str) else model
    ).encode


@dataclass(slots=True, frozen=True)
class SentenceTransformers(EmbeddingFunc[str, NumpyArray]):
    """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

    The weights of a model given by name are loaded on the first call,
    and the texts are encoded in a worker thread.

    Args:
        model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
            or a `SentenceTransformer` model instance.
        batch_size: Number of texts encoded per forward pass.
        show_progress_bar: Whether to show a progress bar.
        precision: Precision of the embeddings.
        truncate_dim: Dimension to truncate the embeddings to, all if None.
        normalize_embeddings: Whether to normalize the embeddings to unit length.
        device: Device of a model given by name, such as `"cpu"`, detected if None.
    """

    model: str | SentenceTransformer
    batch_size: int = 32
    show_progress_bar: bool | None = None
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32"
    truncate_dim: int | None = None
    normalize_embeddings: bool = False
    device: str | None = None
    _encode: Callable[..., Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_encode", lazy(partial(_load, self.model, self.device))
        )

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[NumpyArray]:
        if not values:
            return []

        vecs = await asyncio.to_thread(
            self._encode,
            cast(list[str], values),
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            precision=self.precision,
            truncate_dim=self.truncate_dim,
            normalize_embeddings=self.normalize_embeddings,
        )

        return list(vecs)


__all__ = ["SentenceTransformers"]
