"""sentence-transformers sparse embedding provider."""

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any, cast, override

import sentence_transformers

from ....helpers import lazy
from ....typing import EmbeddingFunc, SparseVector


def _load(model: str | sentence_transformers.SparseEncoder) -> Callable[..., Any]:
    return (
        sentence_transformers.SparseEncoder(model) if isinstance(model, str) else model
    ).encode


@dataclass(slots=True, frozen=True)
class SparseEncoder(EmbeddingFunc[str, SparseVector]):
    """Sparse embeddings using `sentence-transformers <https://www.sbert.net/>`_ SparseEncoder.

    Wraps any `SparseEncoder` model, including SPLADE variants and other
    sparse embedding models.  Produces sparse vectors where each dimension
    corresponds to a vocabulary token and the value represents the token's
    importance.
    The weights of a model given by name are loaded on the first call,
    and the texts are encoded in a worker thread.

    Args:
        model: Either the name of a sparse model (e.g.,
            `"naver/splade-cocondenser-ensembledistil"`) or a
            `SparseEncoder` instance.
        batch_size: Batch size for encoding.
        show_progress_bar: Whether to show a progress bar.
    """

    model: str | sentence_transformers.SparseEncoder
    batch_size: int = 32
    show_progress_bar: bool | None = None
    _encode: Callable[..., Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_encode", lazy(partial(_load, self.model)))

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[SparseVector]:
        if not values:
            return []

        embeddings = await asyncio.to_thread(
            self._encode,
            cast(list[str], values),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
        )

        # Convert 2D sparse COO tensor to list of {token_id: weight}
        coalesced = embeddings.coalesce()
        indices = coalesced.indices()  # [2, nnz]
        values = coalesced.values()  # [nnz]

        result: list[SparseVector] = [{} for _ in range(len(values))]

        for idx in range(indices.shape[1]):
            row = int(indices[0, idx])
            col = int(indices[1, idx])
            val = float(values[idx])
            if val != 0.0:
                result[row][col] = val

        return result


__all__ = ["SparseEncoder"]
