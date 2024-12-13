import functools
import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast, override

import numpy as np
from numpy import typing as npt
from scipy.spatial.distance import cosine as scipy_cosine

from ...helpers import optional_dependencies
from ...typing import (
    BatchSimFunc,
    FilePath,
    HasMetadata,
    JsonDict,
    KeyValueStore,
    SimSeq,
)

type EmbedFunc[T] = Callable[[Sequence[str]], Mapping[str, T]]


class VectorDB[T](KeyValueStore[str, T], Protocol):
    def dump(self) -> None: ...

    def __call__(self, texts: Sequence[str], func: EmbedFunc[T], /) -> dict[str, T]:
        if not self.frozen:
            new_texts = [text for text in texts if text not in self.store]
            self.store.update(func(new_texts))

        return self.store


@dataclass(slots=True)
class numpy_db(VectorDB[npt.NDArray]):
    store: dict[str, npt.NDArray] = field(default_factory=dict, init=False, repr=False)
    path: FilePath | None = None
    frozen: bool = False

    def __post_init__(self) -> None:
        if self.path:
            try:
                self.store = np.load(self.path)
            except Exception:
                pass

    def dump(self) -> None:
        if not self.path:
            raise ValueError("Path not provided")

        np.savez_compressed(self.path, **self.store)


with optional_dependencies():
    import torch

    @dataclass(slots=True)
    class torch_db(VectorDB[torch.Tensor]):
        store: dict[str, torch.Tensor] = field(
            default_factory=dict, init=False, repr=False
        )
        path: FilePath | None = None
        frozen: bool = False

        def __post_init__(self) -> None:
            if self.path:
                try:
                    self.store = torch.load(self.path)
                except Exception:
                    pass

        def dump(self) -> None:
            if not self.path:
                raise ValueError("Path not provided")

            torch.save(self.store, self.path)


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
        db: VectorDB[npt.NDArray] | None

        def __init__(
            self, model: str | Language, db: VectorDB[npt.NDArray] | None = None
        ):
            if isinstance(model, str):
                self.model = spacy_load(model)
            else:
                self.model = model

            self.db = db

        @property
        @override
        def metadata(self) -> JsonDict:
            return {"model": self.model.meta}

        def embed(self, texts: Sequence[str]) -> dict[str, npt.NDArray]:
            with self.model.select_pipes(enable=[]):
                docs_iterator = self.model.pipe(texts)

            return {
                text: cast(npt.NDArray, doc.vector)
                for text, doc in zip(texts, docs_iterator, strict=True)
            }

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq[float]:
            texts = list(itertools.chain.from_iterable(batches))

            if self.db:
                vecs = self.db(texts, self.embed)
                return [_cosine(vecs[x], vecs[y]) for x, y in batches]

            else:
                with self.model.select_pipes(enable=[]):
                    docs_iterator = self.model.pipe(texts)

                docs = dict(zip(texts, docs_iterator, strict=True))

                return [docs[x].similarity(docs[y]) for x, y in batches]


with optional_dependencies():
    import torch
    from sentence_transformers import SentenceTransformer

    @dataclass(slots=True)
    class sentence_transformers(BatchSimFunc[str, float], HasMetadata):
        """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

        Args:
            model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
                or a `SentenceTransformer` model instance.
        """

        model: SentenceTransformer
        db: VectorDB[torch.Tensor] | None
        _metadata: JsonDict

        def __init__(
            self,
            model: str | SentenceTransformer,
            db: VectorDB[torch.Tensor] | None = None,
        ):
            self._metadata = {}

            if isinstance(model, str):
                self.model = SentenceTransformer(model)
                self._metadata["model"] = model
            else:
                self.model = model
                self._metadata["model"] = "custom"

            self.db = db

        @property
        @override
        def metadata(self) -> JsonDict:
            return self._metadata

        def embed(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
            tensor = self.model.encode(cast(list[str], texts), convert_to_tensor=True)

            return dict(zip(texts, tensor, strict=True))

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq[float]:
            if not batches:
                return []

            case_texts, query_texts = zip(*batches, strict=True)

            if self.db:
                vecs = self.db(case_texts + query_texts, self.embed)
                case_tensor = torch.stack([vecs[key] for key in case_texts], dim=0)
                query_tensor = torch.stack([vecs[key] for key in query_texts], dim=0)

            else:
                case_tensor = self.model.encode(case_texts, convert_to_tensor=True)
                query_tensor = self.model.encode(query_texts, convert_to_tensor=True)

            return self.model.similarity_pairwise(case_tensor, query_tensor).tolist()


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
        db: VectorDB[npt.NDArray] | None = None

        def embed(self, texts: Sequence[str]) -> dict[str, npt.NDArray]:
            res = self.client.embeddings.create(
                input=cast(list[str], texts),
                model=self.model,
                encoding_format="float",
            )
            return {
                text: np.array(x.embedding)
                for text, x in zip(texts, res.data, strict=True)
            }

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            texts = list(itertools.chain.from_iterable(batches))
            vecs = self.db(texts, self.embed) if self.db else self.embed(texts)

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
        db: VectorDB[npt.NDArray] | None = None

        def embed(self, texts: Sequence[str]) -> dict[str, npt.NDArray]:
            res = self.client.embed(
                self.model,
                texts,
                truncate=self.truncate,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            vecs = [np.array(x) for x in res["embeddings"]]

            return dict(zip(texts, vecs, strict=True))

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            texts = list(itertools.chain.from_iterable(batches))
            vecs = self.db(texts, self.embed) if self.db else self.embed(texts)

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
        case_db: VectorDB[npt.NDArray] | None = None
        query_db: VectorDB[npt.NDArray] | None = None

        def embed(
            self,
            texts: Sequence[str],
            input_type: Literal["search_document", "search_query"],
        ) -> dict[str, npt.NDArray]:
            res = self.client.v2.embed(
                model=self.model,
                texts=texts,
                input_type=input_type,
                embedding_types="float",
                truncate=self.truncate,
                request_options=self.request_options,
            ).embeddings.float_

            if not res:
                raise ValueError("No embeddings returned")

            return dict(zip(texts, [np.array(x) for x in res], strict=True))

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            if not batches:
                return []

            case_texts, query_texts = zip(*batches, strict=True)
            embed_queries = functools.partial(self.embed, input_type="search_query")
            embed_cases = functools.partial(self.embed, input_type="search_document")

            case_vecs = (
                self.case_db(case_texts, embed_cases)
                if self.case_db
                else embed_cases(case_texts)
            )
            query_vecs = (
                self.query_db(query_texts, embed_queries)
                if self.query_db
                else embed_queries(query_texts)
            )

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
        case_db: VectorDB[npt.NDArray] | None = None
        query_db: VectorDB[npt.NDArray] | None = None

        def embed(
            self, texts: Sequence[str], input_type: Literal["query", "document"]
        ) -> dict[str, npt.NDArray]:
            res = self.client.embed(
                model=self.model,
                texts=cast(list[str], texts),
                input_type=input_type,
                truncation=self.truncation,
            ).embeddings

            return dict(zip(texts, [np.array(x) for x in res], strict=True))

        @override
        def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
            if not batches:
                return []

            case_texts, query_texts = zip(*batches, strict=True)
            embed_queries = functools.partial(self.embed, input_type="query")
            embed_cases = functools.partial(self.embed, input_type="document")

            case_vecs = (
                self.case_db(case_texts, embed_cases)
                if self.case_db
                else embed_cases(case_texts)
            )
            query_vecs = (
                self.query_db(query_texts, embed_queries)
                if self.query_db
                else embed_queries(query_texts)
            )

            return [_cosine(case_vecs[x], query_vecs[y]) for x, y in batches]
