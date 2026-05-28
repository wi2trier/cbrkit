"""Cohere embedding provider."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
from cohere import Client
from cohere.core import RequestOptions

from ....typing import BatchConversionFunc, NumpyArray


@dataclass(slots=True, frozen=True)
class cohere(BatchConversionFunc[str, NumpyArray]):
    """Semantic similarity using Cohere's embedding models

    Args:
        model: Name of the [embedding model](https://docs.cohere.com/reference/embed).
    """

    model: str
    input_type: Literal["search_document", "search_query"] = "search_document"
    client: Client = field(default_factory=Client, repr=False)
    truncate: Literal["NONE", "START", "END"] | None = None
    request_options: RequestOptions | None = None

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        res = self.client.v2.embed(
            model=self.model,
            texts=texts,
            input_type=self.input_type,
            embedding_types="float",
            truncate=self.truncate,
            request_options=self.request_options,
        ).embeddings.float_

        if not res:
            raise ValueError("No embeddings returned")

        return [np.array(x) for x in res]


__all__ = ["cohere"]
