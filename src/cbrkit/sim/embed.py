import asyncio
import itertools
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, override

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

from ..helpers import (
    batchify_conversion,
    batchify_sim,
    chunkify,
    event_loop,
    get_logger,
    optional_dependencies,
)
from ..typing import (
    AnyConversionFunc,
    AnySimFunc,
    BatchConversionFunc,
    BatchSimFunc,
    FilePath,
    Float,
    HasMetadata,
    JsonDict,
    KeyValueStore,
    NumpyArray,
    SimFunc,
    SimSeq,
)

__all__ = [
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

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class cosine(SimFunc[NumpyArray, float]):
    weight: NumpyArray | None = None

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if np.any(u) and np.any(v):
            return 1 - scipy_cosine(u, v, self.weight).__float__()

        return 0.0


@dataclass(slots=True, frozen=True)
class dot(SimFunc[NumpyArray, float]):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return np.dot(u, v).__float__()


@dataclass(slots=True, frozen=True)
class angular(SimFunc[NumpyArray, float]):
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
class euclidean(SimFunc[NumpyArray, float]):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.linalg.norm(u - v).__float__())


@dataclass(slots=True, frozen=True)
class manhattan(SimFunc[NumpyArray, float]):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.sum(np.abs(u - v)).__float__())


default_score_func: SimFunc[NumpyArray, float] = cosine()


@dataclass(slots=True, init=False)
class build[V, S: Float](BatchSimFunc[V, S]):
    """Embedding-based semantic similarity

    Args:
        conversion_func: Embedding function
        sim_func: Similarity score function
        query_conversion_func: Optional query embedding function
    """

    conversion_func: BatchConversionFunc[V, NumpyArray]
    sim_func: BatchSimFunc[NumpyArray, S]
    query_conversion_func: BatchConversionFunc[V, NumpyArray] | None

    def __init__(
        self,
        conversion_func: AnyConversionFunc[V, NumpyArray],
        sim_func: AnySimFunc[NumpyArray, S] = default_score_func,
        query_conversion_func: AnyConversionFunc[V, NumpyArray] | None = None,
    ):
        self.conversion_func = batchify_conversion(conversion_func)
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = (
            batchify_conversion(query_conversion_func)
            if query_conversion_func
            else None
        )

    @override
    def __call__(self, batches: Sequence[tuple[V, V]]) -> SimSeq[S]:
        if not batches:
            return []

        if self.query_conversion_func:
            case_texts, query_texts = zip(*batches, strict=True)
            case_vecs = self.conversion_func(case_texts)
            query_vecs = self.query_conversion_func(query_texts)

            return self.sim_func(list(zip(case_vecs, query_vecs, strict=True)))

        texts = list(itertools.chain.from_iterable(batches))
        vecs = self.conversion_func(texts)

        return self.sim_func([(vecs[i], vecs[i + 1]) for i in range(0, len(vecs), 2)])


@dataclass(slots=True, init=False)
class cache(KeyValueStore[str, NumpyArray]):
    func: BatchConversionFunc[str, NumpyArray] | None
    path: Path | None
    autodump: bool
    store: MutableMapping[str, NumpyArray] = field(repr=False)

    def __init__(
        self,
        func: AnyConversionFunc[str, NumpyArray] | None,
        path: FilePath | None = None,
        autodump: bool = False,
    ):
        self.func = batchify_conversion(func) if func is not None else None
        self.path = Path(path) if isinstance(path, str) else path
        self.autodump = autodump

        if self.path and self.path.exists():
            with np.load(self.path) as data:
                self.store = dict(data)
        else:
            self.store = {}

    def dump(self) -> None:
        if not self.path:
            raise ValueError("Path not provided")

        logger.info(f"Dumping cache to {self.path}")
        np.savez_compressed(self.path, **self.store)
        logger.info("Cache dumped")

    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        new_texts = [text for text in texts if text not in self.store]

        if self.func and new_texts:
            self.store.update(zip(new_texts, self.func(new_texts), strict=True))

            if self.autodump:
                self.dump()

        return [self.store[text] for text in texts]


@dataclass(slots=True, init=False)
class concat[V](BatchConversionFunc[V, NumpyArray]):
    """Concatenated embeddings of multiple models

    Args:
        models: List of embedding models
    """

    embed_funcs: Sequence[BatchConversionFunc[V, NumpyArray]]

    def __init__(
        self,
        funcs: Sequence[AnyConversionFunc[V, NumpyArray]],
    ):
        self.embed_funcs = [batchify_conversion(func) for func in funcs]

    def __call__(self, texts: Sequence[V]) -> Sequence[NumpyArray]:
        nested_vecs = [func(texts) for func in self.embed_funcs]
        return [np.concatenate(vecs, axis=0) for vecs in nested_vecs]


with optional_dependencies():
    from spacy import load as spacy_load
    from spacy.language import Language

    @dataclass(slots=True)
    class spacy(BatchConversionFunc[str, NumpyArray], HasMetadata):
        """Semantic similarity using [spaCy](https://spacy.io/)

        Args:
            model: Either the name of a [spaCy model](https://spacy.io/usage/models)
                or a `spacy.Language` model instance.
        """

        model: Language | Callable[[], Language]

        def __init__(self, model: str | Language | Callable[[], Language]):
            if isinstance(model, str):
                self.model = spacy_load(model)
            else:
                self.model = model

        @property
        @override
        def metadata(self) -> JsonDict:
            return {
                "model": self.model.meta
                if isinstance(self.model, Language)
                else "custom"
            }

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
            if not isinstance(self.model, Language):
                self.model = self.model()

            with self.model.select_pipes(enable=[]):
                docs_iterator = self.model.pipe(texts)

            return [cast(NumpyArray, doc.vector) for doc in docs_iterator]


with optional_dependencies():
    from sentence_transformers import SentenceTransformer

    @dataclass(slots=True)
    class sentence_transformers(BatchConversionFunc[str, NumpyArray], HasMetadata):
        """Semantic similarity using [sentence-transformers](https://www.sbert.net/)

        Args:
            model: Either the name of a [pretrained model](https://www.sbert.net/docs/pretrained_models.html)
                or a `SentenceTransformer` model instance. If a callable is provided, it will be called
                the first time the function is used to create the model instance. This is useful for
                lazy loading of the model.
        """

        model: SentenceTransformer | Callable[[], SentenceTransformer]
        _metadata: JsonDict

        def __init__(
            self, model: str | SentenceTransformer | Callable[[], SentenceTransformer]
        ):
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
        def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
            if not texts:
                return []

            if not isinstance(self.model, SentenceTransformer):
                self.model = self.model()

            return self.model.encode(
                cast(list[str], texts), convert_to_numpy=True
            ).tolist()


with optional_dependencies():
    import tiktoken
    from openai import AsyncOpenAI

    @dataclass(slots=True, frozen=True)
    class openai(BatchConversionFunc[str, NumpyArray]):
        """Semantic similarity using OpenAI's embedding models

        Args:
            model: Name of the [embedding model](https://platform.openai.com/docs/models/embeddings).
        """

        model: str
        client: AsyncOpenAI | Callable[[], AsyncOpenAI] = field(
            default=AsyncOpenAI, repr=False
        )
        chunk_size: int = 2048
        context_size: int = 8192
        truncate: Literal["start", "end"] | None = "end"

        @override
        def __call__(self, batches: Sequence[str]) -> Sequence[NumpyArray]:
            client = (
                self.client if isinstance(self.client, AsyncOpenAI) else self.client()
            )
            return event_loop.get().run_until_complete(
                self.__call_batches__(batches, client)
            )

        async def __call_batches__(
            self, batches: Sequence[str], client: AsyncOpenAI
        ) -> Sequence[NumpyArray]:
            chunk_results = await asyncio.gather(
                *(
                    self.__call_chunk__(chunk, client)
                    for chunk in chunkify(batches, self.chunk_size)
                )
            )

            return list(itertools.chain.from_iterable(chunk_results))

        async def __call_chunk__(
            self,
            batches: Sequence[str],
            client: AsyncOpenAI,
        ) -> Sequence[NumpyArray]:
            res = await client.embeddings.create(
                input=[self.encode(text) for text in batches],
                model=self.model,
                encoding_format="float",
            )
            return [np.array(x.embedding) for x in res.data]

        def encode(self, text: str) -> list[int]:
            value = tiktoken.encoding_for_model(self.model).encode(text)

            if self.truncate == "start":
                return value[-self.context_size :]
            elif self.truncate == "end":
                return value[: self.context_size]

            return value


with optional_dependencies():
    from ollama import Client, Options

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


with optional_dependencies():
    from cohere import Client
    from cohere.core import RequestOptions

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


with optional_dependencies():
    from voyageai import Client  # type: ignore

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
