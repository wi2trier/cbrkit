"""Ollama embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import override

import numpy as np
from ollama import Client, Options

from ....typing import BatchConversionFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class ollama(BatchConversionFunc[str, NumpyArray]):
    """Semantic similarity using Ollama's embedding models

    Args:
        model: Name of the [embedding model]().https://ollama.com/blog/embedding-models
    """

    model: str
    truncate: bool = True
    options: Options | None = None
    keep_alive: float | str | None = None
    client: Client = field(default_factory=Client, repr=False)

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        res = self.client.embed(
            self.model,
            texts,
            truncate=self.truncate,
            options=self.options,
            keep_alive=self.keep_alive,
        )
        return [np.array(x) for x in res["embeddings"]]


__all__ = ["ollama"]
