"""zvec storage backend."""

import asyncio
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from threading import Lock
from typing import Any, cast, override
from uuid import uuid7

import numpy as np
import zvec as zv

from .._util import (
    check_absent,
    check_expected,
    compute_index_diff,
    identity,
    merge_patch,
)
from ..helpers import dist2sim
from ..sim.embed.metrics import Cosine, Dot, Euclidean
from ..typing import (
    Casebase,
    EmbeddingFunc,
    FilePath,
    NumpyArray,
    Results,
    SimMap,
    Storage,
)
from ._common import RowCodec, VectorMetric, make_codec
from ._retriever import (
    IndexType,
    VectorStorageRetriever,
    check_embedder,
    dense_embedder,
    has_sparse,
)

_METRICS: Mapping[
    type[VectorMetric], tuple[zv.MetricType, Callable[[float], float]]
] = {
    Cosine: (zv.MetricType.COSINE, dist2sim),
    # the score of the inner product is the product itself
    Dot: (zv.MetricType.IP, identity),
    Euclidean: (zv.MetricType.L2, dist2sim),
}


@dataclass(slots=True, frozen=True)
class Zvec[K: str, V = str](Storage[K, V]):
    """Zvec storage backend.

    Stores the cases in an embedded zvec collection on disk.  Supports dense
    (vector), sparse (full-text search), and hybrid index types which
    control what data is stored and what indices are built.  The
    `"sparse"` and `"hybrid"` types attach a native full-text-search
    (FTS) index to `value_column`, so lexical retrieval needs
    no sparse embedding model, only the text itself.

    zvec locks a collection for one open handle per process,
    so every operation opens and closes the collection in a worker thread under the lock of the storage,
    and the storage does no I/O until it is used.
    The first write creates the collection with a schema inferred from its cases.
    Use one storage per path, since two of them would compete for the lock of the collection.

    Warning:
        Persisted dense vectors are tied to the `embedder` used when they were written.
        Reopening a collection backed by a different embedding
        model silently returns wrong results when the new model
        has the same dimension, and raises when it does not,
        since `put_index` only re-embeds entries whose text changed.
        Drop the collection (or use a fresh `path`) when changing models.

    Args:
        path: Directory path for the zvec collection.
        table_name: Name of the collection in its schema.
        index_type: Determines what data is stored and which
            indices are created.  `"dense"` stores dense
            embeddings, `"sparse"` builds an FTS index on
            `value_column`, `"hybrid"` does both.
        embedder: Dense embedding function, required for the `"dense"` and `"hybrid"` index types.
        metric: Distance metric for dense vector search.
        value_column: Field holding the embeddable and full-text text.
            With the default `V = str` the casebase value is this
            field, with a `model` it names the model field to embed and index.
        vector_column: Name for the dense vector field.
        fts_tokenizer: Tokenizer used by the FTS index (e.g.
            `"standard"`, or `"jieba"` for Chinese).
        fts_filters: Token filters applied after tokenization (e.g.
            `("lowercase",)`).
        model: A dataclass or pydantic `pydantic.BaseModel`
            describing documents richer than plain text.  When set, `V` is
            the model type: every field becomes a stored scalar field,
            `value_column` names the embeddable field, and reads
            reconstruct model instances.

    Examples:
        >>> import asyncio, tempfile
        >>> storage = Zvec[str, str](f"{tempfile.mkdtemp()}/cases", index_type="sparse")
        >>> asyncio.run(storage.put_index({"a": "hello world", "b": "foo bar"}))
        >>> sorted(asyncio.run(storage.get_index()).items())
        [('a', 'hello world'), ('b', 'foo bar')]
    """

    path: FilePath
    _: KW_ONLY
    table_name: str = "cases"
    index_type: IndexType = "dense"
    embedder: EmbeddingFunc[str, NumpyArray] | None = None
    metric: VectorMetric = field(default_factory=Cosine)
    value_column: str = "value"
    vector_column: str = "vector"
    fts_tokenizer: str = "standard"
    fts_filters: tuple[str, ...] = ("lowercase",)
    model: type[V] | None = None
    _codec: RowCodec[V] = field(init=False, repr=False, compare=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_codec", make_codec(self.model, self.value_column))
        check_embedder(self.index_type, self.embedder)

    async def _use[T](
        self,
        op: Callable[[zv.Collection | None], T],
        schema: zv.CollectionSchema | None = None,
    ) -> T:
        """Run a blocking operation on the opened collection in a worker thread.

        The operation receives None if the collection does not exist and no `schema` to create it is given.
        """

        def run() -> T:
            with self._lock:
                try:
                    collection = zv.open(str(self.path))
                except ValueError:
                    if schema is None:
                        return op(None)

                    collection = zv.create_and_open(str(self.path), schema)
                    self._setup_indices(collection)

                try:
                    return op(collection)
                finally:
                    collection.close()

        return await asyncio.to_thread(run)

    @staticmethod
    def _infer_field_type(value: Any) -> zv.DataType:
        """Infer a zvec DataType from a Python value."""
        if isinstance(value, bool):
            return zv.DataType.BOOL
        if isinstance(value, int):
            return zv.DataType.INT64
        if isinstance(value, float):
            return zv.DataType.DOUBLE
        return zv.DataType.STRING

    def _setup_indices(self, collection: zv.Collection) -> None:
        """Attach the FTS index to `value_column` for sparse/hybrid types."""
        if has_sparse(self.index_type):
            collection.create_index(
                self.value_column,
                zv.FtsIndexParam(
                    tokenizer_name=self.fts_tokenizer,
                    filters=list(self.fts_filters),
                ),
            )

    async def _build_docs(
        self, data: Casebase[K, V]
    ) -> tuple[list[zv.Doc], zv.CollectionSchema]:
        """Build the documents of `data` and the schema of a collection holding them."""
        payloads = [self._codec.encode(value) for value in data.values()]
        vectors: list[dict[str, Any]] = [{} for _ in payloads]
        vector_schemas: list[zv.VectorSchema] = []

        if (embedder := dense_embedder(self.index_type, self.embedder)) is not None:
            metric, _ = _METRICS[type(self.metric)]
            texts = [payload[self.value_column] for payload in payloads]
            embeddings = [np.asarray(vec) for vec in await embedder(texts)]

            for doc_vectors, embedding in zip(vectors, embeddings, strict=True):
                doc_vectors[self.vector_column] = embedding.tolist()

            vector_schemas.append(
                zv.VectorSchema(
                    self.vector_column,
                    zv.DataType.VECTOR_FP32,
                    len(embeddings[0]),
                    index_param=zv.HnswIndexParam(metric_type=metric),
                )
            )

        fields = [zv.FieldSchema(self.value_column, zv.DataType.STRING)]
        fields += [
            zv.FieldSchema(name, self._infer_field_type(value))
            for name, value in payloads[0].items()
            if name != self.value_column
        ]
        docs = [
            zv.Doc(id=str(key), vectors=doc_vectors, fields=payload)
            for key, payload, doc_vectors in zip(data, payloads, vectors, strict=True)
        ]
        schema = zv.CollectionSchema(
            self.table_name, fields=fields, vectors=vector_schemas
        )

        return docs, schema

    def _read(self, collection: zv.Collection | None) -> dict[K, V]:
        """Read every document of the collection, none if it does not exist."""
        if collection is None:
            return {}

        columns = self._codec.columns

        with collection.iter_docs(
            output_fields=list(columns), include_vector=False
        ) as docs:
            return {
                cast(K, doc.id): self._codec.decode(
                    {c: doc.field(c) for c in columns if doc.has_field(c)}
                )
                for doc in docs
            }

    @override
    async def get_index(self) -> Casebase[K, V]:
        """Return the cases of the collection."""
        return await self._use(self._read)

    @override
    async def has_index(self) -> bool:
        """Return whether the collection exists on disk."""
        return await self._use(lambda collection: collection is not None)

    @override
    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Replace the cases of the collection with `data`.

        Only stale or changed entries are deleted or upserted, so unchanged entries skip re-embedding.
        """
        stale, changed = compute_index_diff(await self.get_index(), data)
        await self.patch_index(upsert=changed or None, delete=stale or None)

    @override
    async def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
        add: Sequence[V] = (),
        *,
        insert: Casebase[K, V] | None = None,
        expected: Casebase[K, V] | None = None,
    ) -> list[K]:
        """Apply inserts, replacements, deletes, and additions under new UUIDv7 keys as one mutation.

        The documents are embedded before the mutation, which then runs under the lock of the storage.
        """
        written, delete_keys = merge_patch(upsert, insert, delete)
        keys = cast(list[K], [str(uuid7()) for _ in add])
        written |= dict(zip(keys, add, strict=True))

        if not written and not delete_keys:
            return keys

        docs, schema = await self._build_docs(written) if written else ([], None)

        def write(collection: zv.Collection | None) -> None:
            if expected is not None:
                check_expected(self._read(collection), expected)

            if collection is None:
                return

            if insert:
                check_absent(
                    collection.fetch(
                        list(insert), output_fields=[], include_vector=False
                    ),
                    insert,
                )

            if delete_keys:
                collection.delete([str(key) for key in delete_keys])

            if docs:
                collection.upsert(docs)

            # builds the pending writes into the vector and FTS indices
            collection.optimize()

        await self._use(write, schema)

        return keys


@dataclass(slots=True, frozen=True, kw_only=True)
class ZvecRetriever[K: str](VectorStorageRetriever[K, Zvec[K, Any]]):
    """Search of a `Zvec` storage.

    Several retrievers can share one storage to search it with different settings.
    Sparse search uses the native full-text search over `value_column`,
    and hybrid search fuses both by weighted reciprocal rank fusion.

    Args:
        where: Filter expression applied to every search, which may constrain scalar fields other than `value_column`.
    """

    where: str | None = None

    async def _run(
        self,
        queries: Sequence[str],
        build: Callable[[str, NumpyArray | None], zv.Query],
        score: Callable[[float], float],
        embed: bool,
    ) -> Results[K, str]:
        """Run one zvec query per query text in one operation of the storage."""
        vectors = await self._embed(queries) if embed else [None] * len(queries)
        vc = self.storage.value_column

        def run(collection: zv.Collection | None) -> Results[K, str]:
            if collection is None:
                return [({}, {}) for _ in queries]

            count = collection.stats.doc_count
            # zvec needs a finite and positive topk
            topk = max(1, count if self.limit is None else min(self.limit, count))
            results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

            for query, vector in zip(queries, vectors, strict=True):
                docs = collection.query(
                    build(query, vector),
                    topk=topk,
                    filter=self.where,
                    output_fields=[vc],
                )
                results.append(
                    (
                        {
                            cast(K, doc.id): cast(
                                str, doc.field(vc) if doc.has_field(vc) else ""
                            )
                            for doc in docs
                        },
                        {
                            cast(K, doc.id): score(doc.score)
                            for doc in docs
                            if doc.score is not None
                        },
                    )
                )

            return results

        return await self.storage._use(run)

    def _dense_query(self, vector: NumpyArray | None) -> zv.Query:
        return zv.Query(self.storage.vector_column, vector=np.asarray(vector))

    def _fts_query(self, text: str) -> zv.Query:
        return zv.Query(self.storage.value_column, fts=zv.Fts(match_string=text))

    @override
    async def _search_dense(self, queries: Sequence[str]) -> Results[K, str]:
        _, score = _METRICS[type(self.storage.metric)]

        return await self._run(
            queries, lambda _, vector: self._dense_query(vector), score, embed=True
        )

    @override
    async def _search_sparse(self, queries: Sequence[str]) -> Results[K, str]:
        # FTS scores are relevance-ordered (higher = better)
        return await self._run(
            queries, lambda query, _: self._fts_query(query), identity, embed=False
        )


__all__ = [
    "Zvec",
    "ZvecRetriever",
]
