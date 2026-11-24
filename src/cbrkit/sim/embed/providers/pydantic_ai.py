"""pydantic-ai embedding provider."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import batched, chain
from typing import override

import numpy as np
from pydantic_ai.embeddings import Embedder, EmbeddingSettings, EmbedInputType

from ....helpers import bounded_gather
from ....typing import EmbeddingFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class PydanticAI(EmbeddingFunc[str, NumpyArray]):
    """Embeddings using pydantic-ai's Embedder interface.

    The texts are sent in requests of at most `batch_size` texts, `concurrency` of them at a time.

    Args:
        embedder: The pydantic-ai embedder.
        input_type: Whether the texts are queries or documents, which some models embed differently.
        batch_size: Most texts of one request, OpenAI accepts 2048.
        concurrency: Most requests at a time.
        settings: Passed to every request, such as `truncate`, `dimensions` or `extra_body`.
        max_input_tokens: Cuts longer texts client-side by counting their tokens with the model,
            for providers that do not truncate themselves.
            If None, the model's own limit if it reports one (OpenAI's 8192), else sent as they are.
    """

    embedder: Embedder = field(repr=False)
    input_type: EmbedInputType
    batch_size: int = 2048
    concurrency: int = 4
    settings: EmbeddingSettings | None = None
    max_input_tokens: int | None = None

    @override
    async def __call__(self, values: Sequence[str], /) -> Sequence[NumpyArray]:
        if not values:
            return []

        limit = self.max_input_tokens or await self.embedder.max_input_tokens()

        if limit is not None:
            values = await asyncio.gather(*(self._fit(text, limit) for text in values))

        async def embed(batch: Sequence[str]) -> list[NumpyArray]:
            res = await self.embedder.embed(
                batch, input_type=self.input_type, settings=self.settings
            )

            return [np.array(x) for x in res.embeddings]

        results = await bounded_gather(
            batched(values, self.batch_size), embed, limit=self.concurrency
        )

        return list(chain.from_iterable(results))

    async def _fit(self, text: str, limit: int) -> str:
        """The text cut until the model counts at most `limit` tokens of it."""
        # a token covers at least one byte, so shorter texts fit uncounted
        if len(text.encode()) <= limit:
            return text

        tokens = await self.embedder.count_tokens(text)

        while tokens > limit:
            text = text[: len(text) * limit // tokens]
            tokens = await self.embedder.count_tokens(text)

        return text


__all__ = ["PydanticAI"]
