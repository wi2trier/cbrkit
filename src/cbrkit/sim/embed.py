import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast, override

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

from ..helpers import optional_dependencies
from ..typing import (
    BatchSimFunc,
    FilePath,
    HasMetadata,
    JsonDict,
    KeyValueStore,
    MapConversionFunc,
    NumpyArray,
    SimSeq,
)

__all__ = [
    "EmbedFunc",
    "ScoreFunc",
    "cosine",
    "dot",
    "angular",
    "euclidean",
    "manhattan",
    "build",
    "cache",
    "concat",
    "spacy",
    "sentence_transformers",
    "openai",
    "ollama",
    "cohere",
    "voyageai",
]


class EmbedFunc(MapConversionFunc[str, NumpyArray], Protocol): ...


class ScoreFunc(Protocol):
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float: ...


@dataclass(slots=True, frozen=True)
class cosine(ScoreFunc):
    weight: NumpyArray | None = None

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if np.any(u) and np.any(v):
            return 1 - scipy_cosine(u, v, self.weight).astype(float)

        return 0.0


@dataclass(slots=True, frozen=True)
class dot(ScoreFunc):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return np.dot(u, v).astype(float)


@dataclass(slots=True, frozen=True)
class angular(ScoreFunc):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            try:
                return (
                    1.0
                    - np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                    / np.pi
                )
            except Exception:
                pass

        return 0.0


@dataclass(slots=True, frozen=True)
class euclidean(ScoreFunc):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.linalg.norm(u - v)).astype(float)


@dataclass(slots=True, frozen=True)
class manhattan(ScoreFunc):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.sum(np.abs(u - v)))


default_score_func: ScoreFunc = cosine()


@dataclass(slots=True, frozen=True)
class build(BatchSimFunc[str, float]):
    """Embedding-based semantic similarity

    Args:
        embed_func: Embedding function
        score_func: Similarity score function
        query_embed_func: Optional query embedding function
    """

    embed_func: EmbedFunc
    score_func: ScoreFunc = default_score_func
    query_embed_func: EmbedFunc | None = None

    @override
    def __call__(self, batches: Sequence[tuple[str, str]]) -> SimSeq:
        if not batches:
            return []

        if self.query_embed_func:
            case_texts, query_texts = zip(*batches, strict=True)
            case_vecs = self.embed_func(case_texts)
            query_vecs = self.query_embed_func(query_texts)

            return [self.score_func(case_vecs[x], query_vecs[y]) for x, y in batches]

        texts = list(itertools.chain.from_iterable(batches))
        vecs = self.embed_func(texts)

        return [self.score_func(vecs[x], vecs[y]) for x, y in batches]


@dataclass(slots=True)
class cache(KeyValueStore[str, NumpyArray], EmbedFunc):
    func: MapConversionFunc[str, NumpyArray]
    path: FilePath | None = None
    frozen: bool = False
    store: dict[str, NumpyArray] = field(default_factory=dict, init=False, repr=False)

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

    def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
        if not self.frozen:
            new_texts = [text for text in texts if text not in self.store]
            self.store.update(self.func(new_texts))

        return self.store


@dataclass(slots=True, frozen=True)
class concat(EmbedFunc):
    """Concatenated embeddings of multiple models

    Args:
        models: List of embedding models
    """

    embed_funcs: list[EmbedFunc]

    def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
        vecs = [func(texts) for func in self.embed_funcs]
        return {
            text: np.concatenate([vec[text] for vec in vecs], axis=0) for text in texts
        }


with optional_dependencies():
    from spacy import load as spacy_load
    from spacy.language import Language

    @dataclass(slots=True)
    class spacy(EmbedFunc, HasMetadata):
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
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
            with self.model.select_pipes(enable=[]):
                docs_iterator = self.model.pipe(texts)

            return {
                text: cast(NumpyArray, doc.vector)
                for text, doc in zip(texts, docs_iterator, strict=True)
            }


with optional_dependencies():
    from sentence_transformers import SentenceTransformer

    @dataclass(slots=True)
    class sentence_transformers(EmbedFunc, HasMetadata):
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
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
            if not texts:
                return {}

            vecs = self.model.encode(cast(list[str], texts), convert_to_numpy=True)

            return dict(zip(texts, vecs, strict=True))


with optional_dependencies():
    from openai import OpenAI

    @dataclass(slots=True, frozen=True)
    class openai(EmbedFunc):
        """Semantic similarity using OpenAI's embedding models

        Args:
            model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
        """

        model: str
        client: OpenAI = field(default_factory=OpenAI, repr=False)

        @override
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
            res = self.client.embeddings.create(
                input=cast(list[str], texts),
                model=self.model,
                encoding_format="float",
            )
            return {
                text: np.array(x.embedding)
                for text, x in zip(texts, res.data, strict=True)
            }


with optional_dependencies():
    from ollama import Client, Options

    @dataclass(slots=True, frozen=True)
    class ollama(EmbedFunc):
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
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
            res = self.client.embed(
                self.model,
                texts,
                truncate=self.truncate,
                options=self.options,
                keep_alive=self.keep_alive,
            )
            vecs = [np.array(x) for x in res["embeddings"]]

            return dict(zip(texts, vecs, strict=True))


with optional_dependencies():
    from cohere import Client
    from cohere.core import RequestOptions

    @dataclass(slots=True, frozen=True)
    class cohere(EmbedFunc):
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
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
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

            return dict(zip(texts, [np.array(x) for x in res], strict=True))


with optional_dependencies():
    from voyageai import Client  # type: ignore

    @dataclass(slots=True, frozen=True)
    class voyageai(EmbedFunc):
        """Semantic similarity using Voyage AI's embedding models

        Args:
            model: Name of the [embedding model](https://docs.voyageai.com/docs/embeddings).
        """

        model: str
        input_type: Literal["query", "document"] = "document"
        client: Client = field(default_factory=Client, repr=False)
        truncation: bool = True

        @override
        def __call__(self, texts: Sequence[str]) -> dict[str, NumpyArray]:
            res = self.client.embed(
                model=self.model,
                texts=cast(list[str], texts),
                input_type=self.input_type,
                truncation=self.truncation,
            ).embeddings

            return dict(zip(texts, [np.array(x) for x in res], strict=True))
