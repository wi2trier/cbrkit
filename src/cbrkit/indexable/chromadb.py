"""ChromaDB storage backend."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

import chromadb as cdb
from chromadb.api import ClientAPI

from ..helpers import get_logger
from ..typing import Casebase, IndexableFunc
from ._common import _compute_index_diff, _normalize_patch_keys

logger = get_logger(__name__)


@dataclass(slots=True)
class chromadb[K: str](IndexableFunc[Casebase[K, str], Collection[K]]):
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

    The write methods accept a per-call ``metadata`` keyword argument
    — a ``{key: extras}`` mapping — when callers need to attach extra
    metadata that cannot be derived from ``(key, value)`` at storage
    construction time.
    """

    path: str
    collection_name: str
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    embedding_func: cdb.EmbeddingFunction[Any] | None = None
    sparse_embedding_func: cdb.SparseEmbeddingFunction[Any] | None = None
    sparse_key: str = "sparse_embedding"
    _client: ClientAPI = field(init=False, repr=False)
    _collection: cdb.Collection | None = field(default=None, init=False, repr=False)

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
        data: Casebase[K, str],
        metadata: Mapping[K, cdb.Metadata] | None = None,
    ) -> tuple[list[str], list[str], list[cdb.Metadata] | None]:
        """Prepare IDs, documents, and metadatas from a casebase."""
        ids = [str(k) for k in data.keys()]
        values = list(data.values())
        metadatas: list[cdb.Metadata] | None = None

        if metadata is not None:
            metadatas = [metadata[k] for k in data.keys()]

        return ids, values, metadatas

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
    def index(self) -> Casebase[K, str]:
        """Return the indexed casebase from the ChromaDB collection."""
        if self._collection is None:
            return {}

        result = self._collection.get()
        ids = result["ids"]
        docs = result["documents"] or []
        return {cast(K, id_): doc for id_, doc in zip(ids, docs, strict=True)}

    @override
    def put_index(
        self,
        data: Casebase[K, str],
        *,
        metadata: Mapping[K, cdb.Metadata] | None = None,
    ) -> None:
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
                ids, documents, metadatas = self._prepare_documents(
                    data, metadata
                )
                self._batched_write("add", ids, documents, metadatas)

            return

        existing = self.index
        stale_keys, changed_or_new = _compute_index_diff(existing, data)

        if not stale_keys and not changed_or_new:
            return

        self.patch_index(
            upsert=changed_or_new or None,
            delete=stale_keys or None,
            metadata=metadata,
        )

    @override
    def upsert_index(
        self,
        data: Casebase[K, str],
        *,
        metadata: Mapping[K, cdb.Metadata] | None = None,
    ) -> None:
        """Upsert documents into the ChromaDB collection.

        If no collection exists yet, delegates to :meth:`put_index`.
        """
        if self._collection is None:
            self.put_index(data, metadata=metadata)
            return

        if not data:
            return

        ids, documents, metadatas = self._prepare_documents(data, metadata)
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
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
        *,
        metadata: Mapping[K, cdb.Metadata] | None = None,
    ) -> None:
        """Apply inserts, replacements, and deletes to the ChromaDB collection."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_keys = normalized

        if self._collection is None:
            if upsert:
                self.put_index(upsert, metadata=metadata)
            return

        if delete_keys:
            self._collection.delete(ids=[str(k) for k in delete_keys])

        if upsert:
            ids, documents, metadatas = self._prepare_documents(
                upsert, metadata
            )
            self._batched_write("upsert", ids, documents, metadatas)


__all__ = ["chromadb"]
