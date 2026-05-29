"""ChromaDB storage backend."""

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

import chromadb as cdb
from chromadb.api import ClientAPI

from ..helpers import get_logger
from ..typing import Casebase, IndexableFunc
from ._common import (
    RowCodec,
    _compute_index_diff,
    _normalize_patch_keys,
    make_codec,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class chromadb[K: str, V = str](IndexableFunc[Casebase[K, V], Collection[K]]):
    """ChromaDB storage backend.

    Manages a persistent ChromaDB collection.  Supports dense,
    sparse, and hybrid index types which control what embedding
    functions and schema are configured.

    Warning:
        Persisted vectors are tied to the
        :paramref:`embedding_func` /
        :paramref:`sparse_embedding_func` used when they were
        written.  Reopening a collection backed by a different
        embedding model silently returns wrong results when the
        new model has the same dimension, and raises on `add`
        when it does not — :meth:`put_index` only re-embeds
        entries whose text changed.  Drop the collection (or use
        a fresh `collection_name`) when changing models.

    Args:
        path: Directory for PersistentClient storage.
        collection_name: Collection name.
        index_type: Determines what embeddings and indices are
            configured.  `"dense"` uses the embedding function,
            `"sparse"` uses the sparse embedding function,
            `"hybrid"` uses both.
        embedding_func: ChromaDB `EmbeddingFunction` for dense
            embeddings.  Required for `"dense"` and `"hybrid"`.
        sparse_embedding_func: ChromaDB
            `SparseEmbeddingFunction` for sparse embeddings.
            Required for `"sparse"` and `"hybrid"`.
        sparse_key: Key name for the sparse vector index in the
            ChromaDB schema.
        value_field: Field holding the embeddable text (stored as the Chroma
            *document*).  With the default ``V = str`` the casebase value *is*
            the document; with a *model* it names the model field to embed.
        model: A dataclass or pydantic :class:`~pydantic.BaseModel`
            describing entries richer than plain text.  When set, ``V`` is the
            model type: ``value_field`` becomes the document and every other
            field becomes Chroma metadata, with reads reconstructing model
            instances.  This replaces any side-channel metadata — extra
            fields ride on the typed value itself.
    """

    path: str
    collection_name: str
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    embedding_func: cdb.EmbeddingFunction[Any] | None = None
    sparse_embedding_func: cdb.SparseEmbeddingFunction[Any] | None = None
    sparse_key: str = "sparse_embedding"
    value_field: str = "value"
    model: type[V] | None = None
    _client: ClientAPI = field(init=False, repr=False)
    _collection: cdb.Collection | None = field(default=None, init=False, repr=False)

    @property
    def _codec(self) -> RowCodec[V]:
        return make_codec(self.model, self.value_field)

    def __post_init__(self) -> None:
        if self.index_type in ("dense", "hybrid") and self.embedding_func is None:
            raise ValueError(
                f"embedding_func is required for index_type={self.index_type!r}"
            )
        if (
            self.index_type in ("sparse", "hybrid")
            and self.sparse_embedding_func is None
        ):
            raise ValueError(
                f"sparse_embedding_func is required for index_type={self.index_type!r}"
            )

        self._client = cdb.PersistentClient(path=self.path)

        try:
            self._collection = self._client.get_collection(
                self.collection_name,
                embedding_function=self.embedding_func,
            )
        except Exception as exc:
            logger.debug(
                "Could not open existing collection %r: %s",
                self.collection_name,
                exc,
            )
            self._collection = None

    @override
    def has_index(self) -> bool:
        """Return whether a collection exists."""
        return self._collection is not None

    def _build_schema(self) -> cdb.Schema | None:
        """Build collection schema with sparse vector index if needed."""
        if self.index_type not in ("sparse", "hybrid"):
            return None

        return cdb.Schema().create_index(
            key=self.sparse_key,
            config=cdb.SparseVectorIndexConfig(
                embedding_function=self.sparse_embedding_func,
                source_key=cdb.K.DOCUMENT.name,
            ),
        )

    def _prepare_documents(
        self,
        data: Casebase[K, V],
    ) -> tuple[list[str], list[str], list[cdb.Metadata] | None]:
        """Prepare IDs, documents, and metadatas from a casebase."""
        codec = self._codec
        ids = [str(k) for k in data.keys()]
        documents: list[str] = []
        metadatas: list[cdb.Metadata] = []
        has_extras = False

        for value in data.values():
            payload = codec.encode(value)
            documents.append(payload[self.value_field])
            extra = {k: v for k, v in payload.items() if k != self.value_field}
            metadatas.append(cast(cdb.Metadata, extra))
            has_extras = has_extras or bool(extra)

        return ids, documents, (metadatas if has_extras else None)

    def _batched_write(
        self,
        op: Literal["add", "upsert"],
        ids: list[str],
        documents: list[str],
        metadatas: list[cdb.Metadata] | None,
    ) -> None:
        """Dispatch `add`/`upsert` in chunks sized by the client limit."""
        assert self._collection is not None
        batch_size = self._client.get_max_batch_size()
        fn = self._collection.add if op == "add" else self._collection.upsert

        for start in range(0, len(ids), batch_size):
            stop = start + batch_size
            fn(
                ids=ids[start:stop],
                documents=documents[start:stop],
                metadatas=metadatas[start:stop] if metadatas is not None else None,
            )

    @property
    @override
    def index(self) -> Casebase[K, V]:
        """Return the indexed casebase from the ChromaDB collection."""
        if self._collection is None:
            return {}

        codec = self._codec
        result = self._collection.get()
        ids = result["ids"]
        docs = result["documents"] or []
        metas = result["metadatas"] or [None] * len(ids)
        return {
            cast(K, id_): codec.decode({self.value_field: doc, **(meta or {})})
            for id_, doc, meta in zip(ids, docs, metas, strict=True)
        }

    @override
    def put_index(self, data: Casebase[K, V]) -> None:
        """Replace the ChromaDB collection contents with *data*.

        On first call the collection is created from scratch.  On
        subsequent calls only stale or changed entries are
        deleted/upserted, so unchanged entries skip re-embedding.
        """
        if self._collection is None:
            collection = self._client.create_collection(
                name=self.collection_name,
                schema=self._build_schema(),
                embedding_function=self.embedding_func,
            )

            self._collection = collection

            if data:
                ids, documents, metadatas = self._prepare_documents(data)
                self._batched_write("add", ids, documents, metadatas)

            return

        existing = self.index
        stale_keys, changed_or_new = _compute_index_diff(existing, data)

        if not stale_keys and not changed_or_new:
            return

        self.patch_index(
            upsert=changed_or_new or None,
            delete=stale_keys or None,
        )

    @override
    def upsert_index(self, data: Casebase[K, V]) -> None:
        """Upsert documents into the ChromaDB collection.

        If no collection exists yet, delegates to :meth:`put_index`.
        """
        if self._collection is None:
            self.put_index(data)
            return

        if not data:
            return

        ids, documents, metadatas = self._prepare_documents(data)
        self._batched_write("upsert", ids, documents, metadatas)

    @override
    def delete_index(
        self,
        data: Collection[K],
    ) -> None:
        """Remove documents by ID from the ChromaDB collection."""
        if self._collection is None or not data:
            return

        ids = [str(k) for k in data]

        if ids:
            self._collection.delete(ids=ids)

    @override
    def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        """Apply inserts, replacements, and deletes to the ChromaDB collection."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_keys = normalized

        if self._collection is None:
            if upsert:
                self.put_index(upsert)
            return

        if delete_keys:
            self._collection.delete(ids=[str(k) for k in delete_keys])

        if upsert:
            ids, documents, metadatas = self._prepare_documents(upsert)
            self._batched_write("upsert", ids, documents, metadatas)


__all__ = ["chromadb"]
