"""OpenAI embedding provider."""

import asyncio
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
import tiktoken
from openai import AsyncOpenAI

from ....helpers import chunkify, run_coroutine
from ....typing import BatchConversionFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class openai(BatchConversionFunc[str, NumpyArray]):
    """Semantic similarity using OpenAI's embedding models

    Args:
        model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
    """

    model: str
    client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)
    chunk_size: int = 2048
    context_size: int = 8192
    truncate: Literal["start", "end"] | None = "end"

    @override
    def __call__(self, batches: Sequence[str]) -> Sequence[NumpyArray]:
        return run_coroutine(self.__call_batches__(batches))

    async def __call_batches__(
        self, batches: Sequence[str]
    ) -> Sequence[NumpyArray]:
        chunk_results = await asyncio.gather(
            *(
                self.__call_chunk__(chunk)
                for chunk in chunkify(batches, self.chunk_size)
            )
        )

        return list(itertools.chain.from_iterable(chunk_results))

    async def __call_chunk__(self, batches: Sequence[str]) -> Sequence[NumpyArray]:
        res = await self.client.embeddings.create(
            input=[self.encode(text) for text in batches],
            model=self.model,
            encoding_format="float",
        )
        return [np.array(x.embedding) for x in res.data]

    def encode(self, text: str) -> list[int]:
        """Tokenize text using tiktoken and optionally truncate to context size."""
        value = tiktoken.encoding_for_model(self.model).encode(text)

        if self.truncate == "start":
            return value[-self.context_size :]
        elif self.truncate == "end":
            return value[: self.context_size]

        return value


__all__ = ["openai"]
