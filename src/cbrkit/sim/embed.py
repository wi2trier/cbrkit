import asyncio
import itertools
import sqlite3
from collections import ChainMap
from collections.abc import Collection, Iterator, MutableMapping, Sequence
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
    get_logger,
    optional_dependencies,
    run_coroutine,
)
from ..typing import (
    AnyConversionFunc,
    AnySimFunc,
    BatchConversionFunc,
    BatchSimFunc,
    FilePath,
    Float,
    HasMetadata,
    IndexableFunc,
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
    "pydantic_ai",
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

        case_texts, query_texts = zip(*batches, strict=True)

        if self.query_conversion_func:
            case_vecs = self.conversion_func(case_texts)
            query_vecs = self.query_conversion_func(query_texts)
        else:
            # Batch all texts together when using the same conversion function
            all_texts = list(case_texts) + list(query_texts)
            all_vecs = self.conversion_func(all_texts)
            case_vecs = all_vecs[: len(case_texts)]
            query_vecs = all_vecs[len(case_texts) :]

        return self.sim_func(list(zip(case_vecs, query_vecs, strict=True)))


@dataclass(slots=True, init=False)
class cache(BatchConversionFunc[str, NumpyArray], IndexableFunc[Collection[str]]):
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

    def _compute_vecs(self, texts: Sequence[str] | set[str]) -> dict[str, NumpyArray]:
        new_texts = [text for text in texts if text not in self.store]

        if not new_texts:
            return {}

        if self.func is None:
            raise ValueError("Conversion function is required for computing embeddings")

        new_vecs = self.func(new_texts)

        return dict(zip(new_texts, new_vecs, strict=True))

    @property
    @override
    def index(self) -> Collection[str]:
        """Return the indexed texts."""
        return self.store.keys()

    @override
    def create_index(self, data: Collection[str]) -> None:
        """Rebuild index, reusing existing embeddings where possible."""
        data_set = set(data)

        if not data_set:
            self.store.clear()

            if self.path is not None and self.table is not None:
                with self.connect() as con:
                    con.execute(f'DELETE FROM "{self.table}"')
                    con.commit()

            return

        # Remove entries no longer needed
        stale_keys = set(self.store.keys()) - data_set
        for key in stale_keys:
            del self.store[key]

        # Compute only new embeddings (_compute_vecs skips texts already in store)
        new_vecs = self._compute_vecs(data_set)
        self.store.update(new_vecs)

        if self.path is not None and self.table is not None:
            with self.connect() as con:
                if stale_keys:
                    con.executemany(
                        f'DELETE FROM "{self.table}" WHERE text = ?',
                        [(text,) for text in stale_keys],
                    )

                if new_vecs:
                    con.executemany(
                        f'INSERT INTO "{self.table}" (text, vector) VALUES(?, ?)',
                        [
                            (text, vector.astype(np.float64).tobytes())
                            for text, vector in new_vecs.items()
                        ],
                    )

                con.commit()

    @override
    def update_index(self, data: Collection[str]) -> None:
        """Add new embeddings to an existing index."""
        new_vecs = self._compute_vecs(set(data))
        self.store.update(new_vecs)

        if self.path is not None and self.table is not None and new_vecs:
            with self.connect() as con:
                con.executemany(
                    f'INSERT INTO "{self.table}" (text, vector) VALUES(?, ?)',
                    [
                        (text, vector.astype(np.float64).tobytes())
                        for text, vector in new_vecs.items()
                    ],
                )
                con.commit()

    @override
    def delete_index(self, data: Collection[str]) -> None:
        """Remove specific entries from the store and SQLite."""
        for text in data:
            self.store.pop(text, None)

        if self.path is not None and self.table is not None and data:
            with self.connect() as con:
                con.executemany(
                    f'DELETE FROM "{self.table}" WHERE text = ?',
                    [(text,) for text in data],
                )
                con.commit()

    @override
    def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
        """Compute embeddings for the given texts.

        Texts already in the index are returned from the cache.
        Uncached texts are computed on-the-fly but not persisted;
        use ``index`` to persist embeddings.
        """
        new_vecs = self._compute_vecs(texts)
        tmp_store = ChainMap(new_vecs, self.store)

        return [tmp_store[text] for text in texts]


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
        return [np.concatenate(vecs, axis=0) for vecs in zip(*nested_vecs)]


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
    from pydantic_ai.embeddings import Embedder
    from pydantic_ai.embeddings.result import EmbedInputType

    @dataclass(slots=True, frozen=True)
    class pydantic_ai(BatchConversionFunc[str, NumpyArray]):
        embedder: Embedder = field(repr=False)
        input_type: EmbedInputType

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
            if not texts:
                return []

            res = self.embedder.embed_sync(texts, input_type=self.input_type)

            return [np.array(x) for x in res.embeddings]


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
        truncate_dim: int | None
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
            truncate_dim: int | None = None,
            normalize_embeddings: bool = False,
        ):
            self._metadata = {}
            self.batch_size = batch_size
            self.show_progress_bar = show_progress_bar
            self.precision = precision
            self.truncate_dim = truncate_dim
            self.normalize_embeddings = normalize_embeddings

            if isinstance(model, str):
                self.model = SentenceTransformer(model)  # pyright: ignore
                self._metadata["model"] = model
            else:
                self.model = model
                self._metadata["model"] = (
                    model.model_card_data.model_id
                    or model.model_card_data.base_model
                    or "custom"
                )

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
                truncate_dim=self.truncate_dim,
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
