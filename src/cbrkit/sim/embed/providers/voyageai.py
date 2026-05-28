"""Voyage AI embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast, override

import numpy as np
from voyageai import Client

from ....typing import BatchConversionFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class voyageai(BatchConversionFunc[str, NumpyArray]):
    """Semantic similarity using Voyage AI's embedding models

    Args:
        model: Name of the [embedding model](https://docs.voyageai.com/docs/embeddings).
    """

    model: str
    input_type: Literal["query", "document"] = "document"
    client: Client = field(default_factory=Client, repr=False)
    truncation: bool = True

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        res = self.client.embed(
            model=self.model,
            texts=cast(list[str], texts),
            input_type=self.input_type,
            truncation=self.truncation,
        ).embeddings

        return [np.array(x) for x in res]


__all__ = ["voyageai"]
