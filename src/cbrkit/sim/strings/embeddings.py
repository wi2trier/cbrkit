import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, override

import numpy as np
from numpy import typing as npt
from scipy.spatial.distance import cosine as scipy_cosine

from ...helpers import optional_dependencies
from ...typing import (
    BatchSimFunc,
    HasMetadata,
    JsonDict,
    SimSeq,
)


def _cosine(u: npt.NDArray, v: npt.NDArray) -> float:
    """Cosine similarity between two vectors

    Args:
        u: First vector
        v: Second vector
    """

    if np.any(u) and np.any(v):
        return 1 - scipy_cosine(u, v).astype(float)

    return 0.0


with optional_dependencies():
    from spacy import load as spacy_load
    from spacy.language import Language

    @dataclass(slots=True)
    class spacy(BatchSimFunc[str, float], HasMetadata):
        """Semantic similarity using [spaCy](https://spacy.io/)

        Args:
            model: Either the name of a [spaCy model](https://spacy.io/usage/models)
                or a `spacy.Language` model instance.
        """

        model: Language

        def __init__(self, model: str | Language):
            if isinstance(model, str):
                self.model = spacy_load(model)
            else:
                self.model = model

        @property
        @override
        def metadata(self) -> JsonDict:
            return {"model": self.model.meta}

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq[float]:
            texts = list(itertools.chain.from_iterable(batches))

            with self.model.select_pipes(enable=[]):
                docs_iterator = self.model.pipe(texts)

            docs = dict(zip(texts, docs_iterator, strict=True))

            return [docs[x].similarity(docs[y]) for x, y in batches]


with optional_dependencies():
    from sentence_transformers import SentenceTransformer

    @dataclass(slots=True)
    class sentence_transformers(BatchSimFunc[str, float], HasMetadata):
        """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

        Args:
            model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
                or a `SentenceTransformer` model instance.
        """

        model: SentenceTransformer
        _metadata: JsonDict

        def __init__(self, model: str | SentenceTransformer):
            self._metadata = {}

            if isinstance(model, str):
                self.model = SentenceTransformer(model)
                self._metadata["model"] = model
            else:
                self.model = model
                self._metadata["model"] = "custom"

        @property
        @override
        def metadata(self) -> JsonDict:
            return self._metadata

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq[float]:
            case_texts, query_texts = zip(*batches, strict=True)
            case_vecs = self.model.encode(case_texts, convert_to_tensor=True)
            query_vecs = self.model.encode(query_texts, convert_to_tensor=True)

            return self.model.similarity_pairwise(case_vecs, query_vecs).tolist()


with optional_dependencies():
    from openai import OpenAI

    @dataclass(slots=True, frozen=True)
    class openai(BatchSimFunc[str, float]):
        """Semantic similarity using OpenAI's embedding models

        Args:
            model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
        """

        model: str
        client: OpenAI = field(default_factory=OpenAI, repr=False)

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            texts = list(itertools.chain.from_iterable(batches))
            res = self.client.embeddings.create(
                input=texts,
                model=self.model,
                encoding_format="float",
            )
            np_vecs = [np.array(x.embedding) for x in res.data]
            vecs = dict(zip(texts, np_vecs, strict=True))

            return [_cosine(vecs[x], vecs[y]) for x, y in batches]


with optional_dependencies():
    from ollama import Client, Options

    @dataclass(slots=True, frozen=True)
    class ollama(BatchSimFunc[str, float]):
        """Semantic similarity using Ollama's embedding models

        Args:
            model: Name of the [embedding model](https://ollama.com/blog/embedding-models).
        """

        model: str
        truncate: bool = True
        options: Options | None = None
        keep_alive: float | str | None = None
        client: Client = field(default_factory=Client, repr=False)

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            texts = list(itertools.chain.from_iterable(batches))
            res = self.client.embed(
                self.model,
                texts,
                truncate=self.truncate,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            np_vecs = [np.array(x) for x in res["embeddings"]]
            vecs = dict(zip(texts, np_vecs, strict=True))

            return [_cosine(vecs[x], vecs[y]) for x, y in batches]


with optional_dependencies():
    from cohere import Client
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere(BatchSimFunc[str, float]):
        """Semantic similarity using Cohere's embedding models

        Args:
            model: Name of the [embedding model](https://docs.cohere.com/reference/embed).
        """

        model: str
        client: Client = field(default_factory=Client, repr=False)
        truncate: Literal["NONE", "START", "END"] | None = None
        request_options: RequestOptions | None = None

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            case_texts, query_texts = zip(*batches, strict=True)

            case_raw_vecs = self.client.v2.embed(
                model=self.model,
                texts=case_texts,
                input_type="search_document",
                embedding_types="float",
                truncate=self.truncate,
                request_options=self.request_options,
            ).embeddings.float_
            query_raw_vecs = self.client.v2.embed(
                model=self.model,
                texts=query_texts,
                input_type="search_document",
                embedding_types="float",
                truncate=self.truncate,
                request_options=self.request_options,
            ).embeddings.float_

            assert case_raw_vecs is not None and query_raw_vecs is not None

            case_np_vecs = [np.array(x) for x in case_raw_vecs]
            query_np_vecs = [np.array(x) for x in query_raw_vecs]

            case_vecs = dict(zip(case_texts, case_np_vecs, strict=True))
            query_vecs = dict(zip(query_texts, query_np_vecs, strict=True))

            return [_cosine(case_vecs[x], query_vecs[y]) for x, y in batches]


with optional_dependencies():
    from voyageai import Client  # type: ignore

    @dataclass(slots=True, frozen=True)
    class voyageai(BatchSimFunc[str, float]):
        """Semantic similarity using Voyage AI's embedding models

        Args:
            model: Name of the [embedding model](https://docs.voyageai.com/docs/embeddings).
        """

        model: str
        client: Client = field(default_factory=Client, repr=False)
        truncation: bool = True

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            case_texts, query_texts = zip(*batches, strict=True)

            case_raw_vecs = self.client.embed(
                model=self.model,
                texts=case_texts,
                input_type="document",
                truncation=self.truncation,
            ).embeddings
            query_raw_vecs = self.client.embed(
                model=self.model,
                texts=query_texts,
                input_type="query",
                truncation=self.truncation,
            ).embeddings

            case_np_vecs = [np.array(x) for x in case_raw_vecs]
            query_np_vecs = [np.array(x) for x in query_raw_vecs]

            case_vecs = dict(zip(case_texts, case_np_vecs, strict=True))
            query_vecs = dict(zip(query_texts, query_np_vecs, strict=True))

            return [_cosine(case_vecs[x], query_vecs[y]) for x, y in batches]
