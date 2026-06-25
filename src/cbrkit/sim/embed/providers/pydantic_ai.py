"""pydantic-ai embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

import numpy as np
from pydantic_ai.embeddings import EmbedInputType, Embedder

from ....typing import BatchConversionFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class pydantic_ai(BatchConversionFunc[str, NumpyArray]):
    """Embeddings using pydantic-ai's Embedder interface."""

    embedder: Embedder = field(repr=False)
    input_type: EmbedInputType

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        if not texts:
            return []

        res = self.embedder.embed_sync(texts, input_type=self.input_type)

        return [np.array(x) for x in res.embeddings]


__all__ = ["pydantic_ai"]
