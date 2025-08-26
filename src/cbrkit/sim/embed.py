import asyncio
import itertools
import sqlite3
from collections.abc import Iterator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast, override

import numpy as np

from ..constants import CACHE_DIR
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
    "load_spacy",
    "sentence_transformers",
    "openai",
    "ollama",
    "cohere",
    "voyageai",
]

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class cosine(SimFunc[NumpyArray, float]):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)

            if u_norm == 0.0 or v_norm == 0.0:
                return 0.0

            # [-1, 1]
            cos_val = np.dot(u, v) / (u_norm * v_norm)

            # [0, 1]
            cos_sim = (cos_val + 1.0) / 2.0

            return np.clip(cos_sim, 0.0, 1.0).__float__()

        return 0.0


@dataclass(slots=True, frozen=True)
class dot(SimFunc[NumpyArray, float]):
    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            dot_prod = (np.dot(u, v) + 1.0) / 2.0

            return np.clip(dot_prod, 0.0, 1.0).__float__()

        return 0.0


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
class cache(BatchConversionFunc[str, NumpyArray]):
    func: BatchConversionFunc[str, NumpyArray] | None
    path: Path | None
    table: str | None
    store: MutableMapping[str, NumpyArray] = field(repr=False)

    def __init__(
        self,
        func: AnyConversionFunc[str, NumpyArray] | None,
        path: FilePath | None = None,
        table: str | None = None,
    ):
        self.func = batchify_conversion(func) if func is not None else None
        self.path = Path(path) if isinstance(path, str) else path
        self.table = table
        self.store = {}

        if self.path is not None:
            if self.table is None:
                raise ValueError("Table name must be specified for disk cache")

            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.connect() as con:
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{self.table}" (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text TEXT NOT NULL UNIQUE,
                        vector BLOB NOT NULL
                    ) STRICT
                """)
                con.commit()

                cur = con.execute(f'SELECT text, vector FROM "{self.table}"')

                for text, vector_blob in cur:
                    self.store[text] = np.frombuffer(vector_blob, dtype=np.float64)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        if self.path is None:
            raise ValueError("Path must be set to use the cache")

        con = sqlite3.connect(self.path, autocommit=False)

        try:
            yield con
        finally:
            con.close()

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        # remove store entries and duplicates
        new_texts = list({text for text in texts if text not in self.store})

        if new_texts:
            if self.func:
                new_vectors = self.func(new_texts)

                for text, vector in zip(new_texts, new_vectors, strict=True):
                    self.store[text] = vector

                if self.path is not None:
                    with self.connect() as con:
                        con.executemany(
                            f'INSERT OR IGNORE INTO "{self.table}" (text, vector) VALUES(?, ?)',
                            [
                                (text, vector.astype(np.float64).tobytes())
                                for text, vector in zip(
                                    new_texts, new_vectors, strict=True
                                )
                            ],
                        )
                        con.commit()

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
    import spacy as spacylib
    from spacy.cli.download import get_latest_version, get_model_filename
    from spacy.language import Language

    def load_spacy(name: str | None, cache_dir: Path = CACHE_DIR) -> Language:
        import tarfile
        import urllib.request

        from rich.progress import Progress, TaskID

        @dataclass(slots=True)
        class ProgressHook(AbstractContextManager):
            description: str
            progress: Progress = field(default_factory=Progress, init=False)
            task: TaskID | None = field(default=None, init=False)

            def __enter__(self):
                self.progress.start()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                self.progress.stop()

            def __call__(self, block_num: int, block_size: int, total_size: int):
                if self.task is None:
                    self.task = self.progress.add_task(
                        self.description, total=total_size
                    )

                downloaded = block_num * block_size

                if downloaded < total_size:
                    self.progress.update(self.task, completed=downloaded)

                if self.progress.finished:
                    self.task = None

        def tarfile_members(
            tf: tarfile.TarFile, prefix: str
        ) -> Iterator[tarfile.TarInfo]:
            prefix_len = len(prefix)

            for member in tf.getmembers():
                if member.path.startswith(prefix):
                    member.path = member.path[prefix_len:]

                    yield member

        if not name:
            return spacylib.blank("en")

        version = get_latest_version(name)
        filename = get_model_filename(name, version, sdist=True)
        versioned_name = f"{name}-{version}"
        cache_file = cache_dir / "spacy" / versioned_name
        tmpfile = cache_file.with_suffix(".tar.gz")

        if not cache_file.exists():
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            download_url = f"{spacylib.about.__download_url__}/{filename}"

            with ProgressHook(
                f"Downloading '{versioned_name}' to '{cache_file.parent}'..."
            ) as hook:
                urllib.request.urlretrieve(download_url, tmpfile, hook)

            with tarfile.open(tmpfile, mode="r:gz") as tf:
                member_prefix = f"{versioned_name}/{name}/{versioned_name}/"
                members = tarfile_members(tf, member_prefix)
                tf.extractall(path=cache_file, members=members)

            tmpfile.unlink()

        return spacylib.load(cache_file)

    @dataclass(slots=True)
    class spacy(BatchConversionFunc[str, NumpyArray], HasMetadata):
        """Semantic similarity using [spaCy](https://spacy.io/)

        Args:
            model: Either the name of a [spaCy model](https://spacy.io/usage/models)
                or a `spacy.Language` model instance.
        """

        model: Language

        def __init__(self, model: str | Language):
            if isinstance(model, str):
                self.model = load_spacy(model)
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
                or a `SentenceTransformer` model instance.
        """

        model: SentenceTransformer
        batch_size: int
        show_progress_bar: bool | None
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"]
        normalize_embeddings: bool
        _metadata: JsonDict

        def __init__(
            self,
            model: str | SentenceTransformer,
            batch_size: int = 32,
            show_progress_bar: bool | None = None,
            precision: Literal[
                "float32", "int8", "uint8", "binary", "ubinary"
            ] = "float32",
            normalize_embeddings: bool = False,
        ):
            self._metadata = {}
            self.batch_size = batch_size
            self.show_progress_bar = show_progress_bar
            self.precision = precision
            self.normalize_embeddings = normalize_embeddings

            if isinstance(model, str):
                self.model = SentenceTransformer(model)  # pyright: ignore
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

            vecs = self.model.encode(
                cast(list[str], texts),
                convert_to_numpy=True,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                precision=self.precision,
                normalize_embeddings=self.normalize_embeddings,
            )

            return list(vecs)


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
        client: AsyncOpenAI = field(default_factory=AsyncOpenAI, repr=False)
        chunk_size: int = 2048
        context_size: int = 8192
        truncate: Literal["start", "end"] | None = "end"

        @override
        def __call__(self, batches: Sequence[str]) -> Sequence[NumpyArray]:
            return event_loop.get().run_until_complete(self.__call_batches__(batches))

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
