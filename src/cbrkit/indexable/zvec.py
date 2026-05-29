"""zvec storage backend."""

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, override

import numpy as np
import zvec as zv  # pyright: ignore[reportMissingImports]  # type: ignore[unresolved-import]

from ..helpers import get_logger
from ..typing import (
    BatchConversionFunc,
    Casebase,
    IndexableFunc,
    NumpyArray,
    SparseVector,
)
from ._common import (
    RowCodec,
    _compute_index_diff,
    _normalize_patch_keys,
    make_codec,
)

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _ZvecCasebaseView[K: str, V](Mapping[K, V]):
    """Lazy mapping backed by a zvec collection.

    Keys are tracked in-memory; values are fetched on demand
    via `Collection.fetch()` and reconstructed via the storage codec.
    """

    _collection: zv.Collection
    _codec: RowCodec[V]
    _payload_fields: tuple[str, ...]
    _keys: frozenset[K]

    def __getitem__(self, key: K) -> V:
        if key not in self._keys:
            raise KeyError(key)
        result = self._collection.fetch(key)
        if key not in result:
            raise KeyError(key)
        doc = result[key]
        payload = {f: doc.field(f) for f in self._payload_fields if doc.has_field(f)}
        return self._codec.decode(payload)

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __hash__(self) -> int:
        return id(self)


@dataclass(slots=True)
class zvec[K: str, V = str](IndexableFunc[Casebase[K, V], Collection[K]]):
    """Zvec storage backend.

    Manages an embedded zvec collection on disk.  Supports dense
    (vector), sparse (sparse vector), and hybrid index types which
    control what data is stored and what indices are built.

    Warning:
        Persisted vectors are tied to the
        :paramref:`conversion_func` /
        :paramref:`sparse_conversion_func` used when they were
        written.  Reopening a collection backed by a different
        embedding model silently returns wrong results when the
        new model has the same dimension, and raises when it does
        not — :meth:`put_index` only re-embeds entries whose text
        changed.  Drop the collection (or use a fresh
        `collection_name`) when changing models.

    Args:
        path: Directory path for the zvec collection.
        collection_name: Collection name used in the schema.
        index_type: Determines what vectors are stored and which
            indices are created.  `"dense"` stores dense
            embeddings, `"sparse"` stores sparse embeddings,
            `"hybrid"` stores both.
        conversion_func: Dense embedding function.  Required for
            `"dense"` and `"hybrid"` index types.
        sparse_conversion_func: Sparse embedding function returning
            `SparseVector` per document.  Required for
            `"sparse"` and `"hybrid"` index types.
        metric_type: Distance metric for dense vector search.
        value_field: Field holding the embeddable text.  With the default
            ``V = str`` the casebase value *is* this field; with a *model*
            it names the model field to embed.
        dense_vector_name: Name for the dense vector field.
        sparse_vector_name: Name for the sparse vector field.
        model: A dataclass or pydantic :class:`~pydantic.BaseModel`
            describing documents richer than plain text.  When set, ``V`` is
            the model type: every field becomes a stored scalar field,
            ``value_field`` names the embeddable field, and reads
            reconstruct model instances.  This replaces any side-channel
            metadata — extra fields ride on the typed value itself, so the
            schema is fixed by the model rather than inferred per batch.
    """

    path: str
    collection_name: str
    index_type: Literal["dense", "sparse", "hybrid"] = "dense"
    conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
    sparse_conversion_func: BatchConversionFunc[str, SparseVector] | None = None
    metric_type: Literal["cosine", "ip", "l2"] = "cosine"
    value_field: str = "value"
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "sparse"
    model: type[V] | None = None
    _collection: zv.Collection | None = field(default=None, init=False, repr=False)
    _keys: set[K] | None = field(default=None, init=False, repr=False)

    @property
    def _codec(self) -> RowCodec[V]:
        return make_codec(self.model, self.value_field)

    def __post_init__(self) -> None:
        if self.index_type in ("dense", "hybrid") and self.conversion_func is None:
            raise ValueError(
                f"conversion_func is required for index_type={self.index_type!r}"
            )
        if (
            self.index_type in ("sparse", "hybrid")
            and self.sparse_conversion_func is None
        ):
            raise ValueError(
                f"sparse_conversion_func is required for index_type={self.index_type!r}"
            )

        try:
            self._collection = zv.open(self.path)
        except Exception:
            self._collection = None

    @property
    def _zvec_metric(self) -> zv.MetricType:
        """Map string metric type to zvec enum."""
        match self.metric_type:
            case "cosine":
                return zv.MetricType.COSINE
            case "ip":
                return zv.MetricType.IP
            case "l2":
                return zv.MetricType.L2

    @override
    def has_index(self) -> bool:
        """Return whether a collection exists on disk."""
        return self._collection is not None

    def search_limit(self) -> int | None:
        """Return the total number of indexed documents, or `None` when empty."""
        if self._keys is None:
            return None

        return len(self._keys)

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

    def _build_schema(self, data: Casebase[K, V]) -> zv.CollectionSchema:
        """Build a CollectionSchema, inferring scalar fields and vector dim."""
        if not data and self.index_type in ("dense", "hybrid"):
            raise ValueError(
                "Cannot build dense/hybrid schema without data to infer dimension."
            )

        sample = self._codec.encode(next(iter(data.values()))) if data else {}
        fields = [zv.FieldSchema(self.value_field, zv.DataType.STRING)]
        fields += [
            zv.FieldSchema(name, self._infer_field_type(value))
            for name, value in sample.items()
            if name != self.value_field
        ]
        vectors: list[zv.VectorSchema] = []

        if self.index_type in ("dense", "hybrid"):
            assert self.conversion_func is not None
            text = sample[self.value_field]
            dimension = len(np.asarray(self.conversion_func([text])[0]))
            vectors.append(
                zv.VectorSchema(
                    self.dense_vector_name,
                    zv.DataType.VECTOR_FP32,
                    dimension,
                    index_param=zv.HnswIndexParam(metric_type=self._zvec_metric),
                )
            )

        if self.index_type in ("sparse", "hybrid"):
            vectors.append(
                zv.VectorSchema(
                    self.sparse_vector_name,
                    zv.DataType.SPARSE_VECTOR_FP32,
                    0,
                )
            )

        return zv.CollectionSchema(
            self.collection_name, fields=fields, vectors=vectors
        )

    def _build_docs(self, casebase: Casebase[K, V]) -> list[zv.Doc]:
        """Build zvec Doc objects from a casebase."""
        codec = self._codec
        keys = list(casebase.keys())
        payloads = [codec.encode(v) for v in casebase.values()]
        texts = [p[self.value_field] for p in payloads]

        dense_vecs = None
        sparse_vecs = None

        if self.index_type in ("dense", "hybrid"):
            assert self.conversion_func is not None
            dense_vecs = self.conversion_func(texts)

        if self.index_type in ("sparse", "hybrid"):
            assert self.sparse_conversion_func is not None
            sparse_vecs = self.sparse_conversion_func(texts)

        docs: list[zv.Doc] = []

        for i, (key, doc_fields) in enumerate(zip(keys, payloads, strict=True)):
            doc_vectors: dict[str, Any] = {}

            if dense_vecs is not None:
                doc_vectors[self.dense_vector_name] = np.asarray(
                    dense_vecs[i]
                ).tolist()

            if sparse_vecs is not None:
                doc_vectors[self.sparse_vector_name] = sparse_vecs[i]

            docs.append(zv.Doc(id=str(key), vectors=doc_vectors, fields=doc_fields))

        return docs

    @property
    @override
    def index(self) -> Casebase[K, V]:
        """Return the indexed casebase."""
        if self._keys is None or self._collection is None:
            return {}
        codec = self._codec
        return _ZvecCasebaseView(
            self._collection,
            codec,
            codec.columns,
            frozenset(self._keys),
        )

    @override
    def put_index(self, data: Casebase[K, V]) -> None:
        """Replace the zvec collection contents with *data*.

        On first call the collection is created from scratch.  On
        subsequent calls only stale or changed entries are
        deleted/upserted, so unchanged entries skip re-embedding.
        """
        if self._collection is None or self._keys is None:
            if self._collection is not None:
                self._collection.destroy()
                self._collection = None

            if not data:
                self._keys = set()
                return

            schema = self._build_schema(data)
            collection = zv.create_and_open(self.path, schema)
            docs = self._build_docs(data)
            collection.insert(docs)

            self._collection = collection
            self._keys = set(data.keys())
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
        """Upsert documents into the zvec collection.

        If no collection exists yet, delegates to :meth:`put_index`.
        """
        if self._collection is None:
            self.put_index(data)
            return

        if not data:
            return

        docs = self._build_docs(data)
        self._collection.upsert(docs)

        if self._keys is not None:
            self._keys.update(data.keys())

    @override
    def delete_index(
        self,
        data: Collection[K],
    ) -> None:
        """Remove documents by ID from the zvec collection."""
        if self._collection is None or not data:
            return

        ids = [str(k) for k in data]

        if ids:
            self._collection.delete(ids)

        if self._keys is not None:
            self._keys -= set(data)

    @override
    def patch_index(
        self,
        upsert: Casebase[K, V] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        """Apply inserts, replacements, and deletes to the zvec collection."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_keys = normalized

        if delete_keys:
            self.delete_index(delete_keys)

        if upsert:
            self.upsert_index(upsert)


__all__ = ["zvec"]
