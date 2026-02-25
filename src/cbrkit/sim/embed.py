import asyncio
import itertools
import sqlite3
from collections import ChainMap
from collections.abc import Collection, Iterator, MutableMapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, override

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
    SparseVector,
)

__all__ = [
    "cosine",
    "dot",
    "angular",
    "euclidean",
    "manhattan",
    "sparse_dot",
    "sparse_cosine",
    "build",
    "cache",
    "concat",
    "spacy",
    "load_spacy",
    "sentence_transformers",
    "sparse_encoder",
    "bm25",
    "pydantic_ai",
    "openai",
    "ollama",
    "cohere",
    "voyageai",
]

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class cosine(SimFunc[NumpyArray, float]):
    """Cosine similarity for dense vectors, normalized to [0, 1]."""

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
    """Dot product similarity for dense vectors, normalized to [0, 1]."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        if u.any() and v.any():
            dot_prod = (np.dot(u, v) + 1.0) / 2.0

            return np.clip(dot_prod, 0.0, 1.0).__float__()

        return 0.0


@dataclass(slots=True, frozen=True)
class angular(SimFunc[NumpyArray, float]):
    """Angular similarity for dense vectors."""

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
    """Euclidean distance-based similarity for dense vectors."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.linalg.norm(u - v).__float__())


@dataclass(slots=True, frozen=True)
class manhattan(SimFunc[NumpyArray, float]):
    """Manhattan distance-based similarity for dense vectors."""

    @override
    def __call__(self, u: NumpyArray, v: NumpyArray) -> float:
        return 1 / (1 + np.sum(np.abs(u - v)).__float__())


@dataclass(slots=True, frozen=True)
class sparse_dot(SimFunc[SparseVector, float]):
    """Dot product similarity for sparse vectors.

    Computes the dot product over shared dimensions and normalizes
    to [0, 1].  Returns 0.0 for empty vectors.

    Examples:
        >>> sparse_dot()({0: 1.0, 1: 2.0}, {0: 3.0, 1: 4.0})
        1.0
        >>> sparse_dot()({}, {0: 1.0})
        0.0
    """

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())
        normalized = (dot_prod + 1.0) / 2.0

        return max(0.0, min(1.0, normalized))


@dataclass(slots=True, frozen=True)
class sparse_cosine(SimFunc[SparseVector, float]):
    """Cosine similarity for sparse vectors.

    Computes cosine similarity between two sparse vectors and normalizes
    to [0, 1].  Returns 0.0 for empty vectors or zero norms.

    Examples:
        >>> sparse_cosine()({0: 1.0, 1: 0.0}, {0: 1.0, 1: 0.0})
        1.0
        >>> sparse_cosine()({0: 1.0}, {1: 1.0})
        0.5
        >>> sparse_cosine()({}, {0: 1.0})
        0.0
    """

    @override
    def __call__(self, u: SparseVector, v: SparseVector) -> float:
        if not u or not v:
            return 0.0

        dot_prod = sum(u[idx] * v[idx] for idx in u.keys() & v.keys())

        u_norm = sum(val * val for val in u.values()) ** 0.5
        v_norm = sum(val * val for val in v.values()) ** 0.5

        if u_norm == 0.0 or v_norm == 0.0:
            return 0.0

        cos_val = dot_prod / (u_norm * v_norm)
        cos_sim = (cos_val + 1.0) / 2.0

        return max(0.0, min(1.0, cos_sim))


default_score_func: SimFunc[NumpyArray, float] = cosine()


@dataclass(slots=True, init=False)
class build[V, E, S: Float](BatchSimFunc[V, S]):
    """Embedding-based semantic similarity.

    Generic over embedding type `E`, supporting both dense
    (`NumpyArray`) and sparse (`SparseVector`) embeddings.

    Args:
        conversion_func: Embedding function producing type `E`.
        sim_func: Similarity score function for type `E`.
        query_conversion_func: Optional query embedding function.
    """

    conversion_func: BatchConversionFunc[V, E]
    sim_func: BatchSimFunc[E, S]
    query_conversion_func: BatchConversionFunc[V, E] | None

    def __init__(
        self,
        conversion_func: AnyConversionFunc[V, E],
        sim_func: AnySimFunc[E, S] = default_score_func,  # type: ignore[assignment]
        query_conversion_func: AnyConversionFunc[V, E] | None = None,
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
class cache(
    BatchConversionFunc[str, NumpyArray],
    IndexableFunc[Collection[str]],
):
    """Embedding cache with optional SQLite persistence."""

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
        """Open a context-managed SQLite connection to the cache database."""
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

    @override
    def has_index(self) -> bool:
        """Return whether the cache contains any entries."""
        return bool(self.store)

    @property
    @override
    def index(self) -> Collection[str]:
        """Return the indexed texts."""
        return self.store.keys()

    @override
    def create_index(self, data: Collection[str]) -> None:
        """Ensure the embedding cache exists and sync it with *data*.

        On first call the cache is populated from scratch.  On
        subsequent calls existing entries are diffed against *data*
        and only stale or new entries are removed/added via
        :meth:`delete_index` and :meth:`update_index`.
        """
        data_set = set(data)

        if not data_set:
            self.delete_index(list(self.store.keys()))
            return

        stale_keys = set(self.store.keys()) - data_set
        new_keys = data_set - set(self.store.keys())

        if not stale_keys and not new_keys:
            return

        if stale_keys:
            self.delete_index(stale_keys)

        if new_keys:
            self.update_index(new_keys)

    @override
    def update_index(self, data: Collection[str]) -> None:
        """Add new embeddings to the cache.

        Texts already present in the cache are skipped.
        """
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
        """Remove specific entries from the cache and SQLite."""
        if not data:
            return

        for text in data:
            self.store.pop(text, None)

        if self.path is not None and self.table is not None:
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
        use `index` to persist embeddings.
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
        """Load a spaCy model by name, downloading it if necessary."""
        import tarfile
        import urllib.request

        from rich.progress import Progress, TaskID

        @dataclass(slots=True)
        class ProgressHook(AbstractContextManager[Any]):
            """Progress reporting hook for URL downloads."""

            description: str
            progress: Progress = field(default_factory=Progress, init=False)
            task: TaskID | None = field(default=None, init=False)

            def __enter__(self):
                super().__enter__()
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
            """Yield tar members with the given prefix stripped from their paths."""
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
            """Return metadata describing the spaCy model."""
            return {
                "model": self.model.meta
                if isinstance(self.model, Language)
                else "custom"
            }

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[NumpyArray]:
            with self.model.select_pipes(enable=[]):
                docs_iterator = self.model.pipe(texts)

            return [np.asarray(doc.vector, dtype=np.float64) for doc in docs_iterator]


with optional_dependencies():
    from pydantic_ai.embeddings import Embedder
    from pydantic_ai.embeddings.result import EmbedInputType

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
                self.model = SentenceTransformer(model)
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
            """Return metadata describing the sentence-transformers model."""
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
    from sentence_transformers.sparse_encoder import SparseEncoder

    @dataclass(slots=True)
    class sparse_encoder(BatchConversionFunc[str, SparseVector], HasMetadata):
        """Sparse embeddings using `sentence-transformers <https://www.sbert.net/>`_ SparseEncoder.

        Wraps any `SparseEncoder` model, including SPLADE variants and other
        sparse embedding models.  Produces sparse vectors where each dimension
        corresponds to a vocabulary token and the value represents the token's
        importance.

        Args:
            model: Either the name of a sparse model (e.g.,
                `"naver/splade-cocondenser-ensembledistil"`) or a
                `SparseEncoder` instance.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to show a progress bar.
        """

        model: SparseEncoder
        batch_size: int
        show_progress_bar: bool | None
        _metadata: JsonDict

        def __init__(
            self,
            model: str | SparseEncoder,
            batch_size: int = 32,
            show_progress_bar: bool | None = None,
        ):
            self._metadata = {}
            self.batch_size = batch_size
            self.show_progress_bar = show_progress_bar

            if isinstance(model, str):
                self.model = SparseEncoder(model)
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
            """Return metadata describing the sparse encoder model."""
            return self._metadata

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[SparseVector]:
            if not texts:
                return []

            embeddings = cast(
                Any,
                self.model.encode(
                    cast(list[str], texts),
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_sparse_tensor=True,
                ),
            )

            # Convert 2D sparse COO tensor to list of {token_id: weight}
            coalesced = embeddings.coalesce()
            indices = coalesced.indices()  # [2, nnz]
            values = coalesced.values()  # [nnz]

            result: list[SparseVector] = [{} for _ in range(len(texts))]

            for idx in range(indices.shape[1]):
                row = int(indices[0, idx])
                col = int(indices[1, idx])
                val = float(values[idx])
                if val != 0.0:
                    result[row][col] = val

            return result


with optional_dependencies():
    import bm25s
    import Stemmer  # type: ignore[import-untyped]
    from bm25s.tokenization import Tokenized

    @dataclass(slots=True)
    class bm25(
        BatchConversionFunc[str, SparseVector],
        IndexableFunc[Collection[str]],
    ):
        """BM25-based sparse embeddings using
        `bm25s <https://github.com/xhluca/bm25s>`_.

        Produces sparse vectors where each dimension corresponds to a
        vocabulary token and the value represents the term frequency.
        Requires fitting on a corpus via `create_index` before use.

        Args:
            language: Language for stemming and stopwords.
            stopwords: Stopword configuration.  `None` uses the
                language default, a `str` sets the stopwords language
                independently, and a `list[str]` provides custom
                stopwords.
        """

        language: str = "english"
        stopwords: str | list[str] | None = None
        _corpus: list[str] | None = field(default=None, init=False, repr=False)
        _retriever: bm25s.BM25 | None = field(default=None, init=False, repr=False)

        @property
        def _stopwords(self) -> str | list[str]:
            return self.stopwords if self.stopwords is not None else self.language

        @property
        def _stemmer(self) -> Stemmer.Stemmer:
            return Stemmer.Stemmer(self.language)

        def _build_retriever(self, texts: Collection[str]) -> bm25s.BM25:
            tokens = bm25s.tokenize(
                list(texts),
                stemmer=self._stemmer,
                stopwords=self._stopwords,
            )
            retriever = bm25s.BM25()
            retriever.index(tokens)
            return retriever

        @override
        def has_index(self) -> bool:
            """Return whether a BM25 corpus has been indexed."""
            return self._corpus is not None

        @property
        @override
        def index(self) -> Collection[str]:
            """Return the indexed corpus or an empty list if not indexed."""
            if self._corpus is None:
                return []
            return self._corpus

        @override
        def create_index(self, data: Collection[str]) -> None:
            """Build a new BM25 index from the given corpus."""
            self._corpus = list(data)
            if self._corpus:
                self._retriever = self._build_retriever(self._corpus)
            else:
                self._retriever = None

        @override
        def update_index(self, data: Collection[str]) -> None:
            """Add new documents to the existing BM25 index."""
            if self._corpus is None:
                self.create_index(data)
                return
            self._corpus.extend(data)
            self._retriever = self._build_retriever(self._corpus)

        @override
        def delete_index(self, data: Collection[str]) -> None:
            """Remove the specified documents from the BM25 index."""
            if self._corpus is None:
                return
            remove_set = set(data)
            self._corpus = [t for t in self._corpus if t not in remove_set]
            if self._corpus:
                self._retriever = self._build_retriever(self._corpus)
            else:
                self._retriever = None

        @override
        def __call__(self, texts: Sequence[str]) -> Sequence[SparseVector]:
            if not texts:
                return []
            if self._retriever is None:
                raise ValueError(
                    "BM25 model must be fitted first. Call create_index()."
                )

            tokenized = cast(
                Tokenized,
                bm25s.tokenize(
                    list(texts),
                    stemmer=self._stemmer,
                    stopwords=self._stopwords,
                ),
            )
            corpus_vocab = self._retriever.vocab_dict
            query_reverse = {v: k for k, v in tokenized.vocab.items()}
            result: list[SparseVector] = []

            for token_ids in tokenized.ids:
                sparse_vec: SparseVector = {}
                for tid in token_ids:
                    token_str = query_reverse.get(int(tid))
                    if token_str is not None and token_str in corpus_vocab:
                        corpus_id = corpus_vocab[token_str]
                        sparse_vec[corpus_id] = sparse_vec.get(corpus_id, 0.0) + 1.0
                result.append(sparse_vec)

            return result


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
            """Tokenize text using tiktoken and optionally truncate to context size."""
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
    from voyageai import Client

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
