"""Pure storage backends for indexable retrieval.

This module provides storage classes that implement
:class:`~cbrkit.typing.IndexableFunc` without any retrieval logic.
Each backend manages database connections, data ingestion, and index
maintenance.  Retriever wrappers in :mod:`cbrkit.retrieval` consume
these storage instances to perform search queries.

Example:
    Create a shared LanceDB storage and attach multiple retrievers::

        import cbrkit

        storage = cbrkit.indexable.lancedb(
            uri="./db",
            table="cases",
            index_type="hybrid",
            conversion_func=embed_func,
        )
        storage.create_index(casebase)

        dense_retriever = cbrkit.retrieval.lancedb(storage=storage, search_type="dense")
        sparse_retriever = cbrkit.retrieval.lancedb(storage=storage, search_type="sparse")
"""

from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from .helpers import get_logger, optional_dependencies
from .typing import BatchConversionFunc, Casebase, IndexableFunc, NumpyArray

__all__ = [
    "chromadb",
    "lancedb",
    "zvec",
]

logger = get_logger(__name__)


def _compute_index_diff[K](
    existing: Casebase[K, str],
    data: Casebase[K, str],
) -> tuple[set[K], Casebase[K, str]]:
    """Compute stale keys and changed/new entries for index updates."""
    new_keys = set(data.keys())
    old_keys = set(existing.keys())
    stale_keys = old_keys - new_keys
    changed_or_new: Casebase[K, str] = {
        k: data[k]
        for k in new_keys
        if k not in existing or existing[k] != data[k]
    }
    return stale_keys, changed_or_new


with optional_dependencies():
    import lancedb as ldb
    import numpy as np

    @dataclass(slots=True)
    class lancedb[K: int | str](
        IndexableFunc[Casebase[K, str], Collection[K]]
    ):
        """LanceDB storage backend.

        Manages an embedded LanceDB database on disk.  Supports dense
        (vector), sparse (FTS/BM25), and hybrid index types which
        control what data is stored and what indices are built.

        Args:
            uri: Path to the LanceDB database directory.
            table: Table name within the database.
            index_type: Determines what data is stored and which
                indices are created.  ``"dense"`` stores embeddings,
                ``"sparse"`` builds an FTS index, ``"hybrid"`` does
                both.
            conversion_func: Embedding function.  Required for
                ``"dense"`` and ``"hybrid"`` index types.
            key_column: Column name for case keys.
            value_column: Column name for case text values.
            vector_column: Column name for dense embedding vectors.
            metadata_func: Optional callable that produces extra
                columns for each row.  Called with ``(key, value)``
                and must return a dict mapping column names to values.
        """

        uri: str
        table: str
        index_type: Literal["dense", "sparse", "hybrid"] = "dense"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        key_column: str = "key"
        value_column: str = "value"
        vector_column: str = "vector"
        metadata_func: Callable[[K, str], dict[str, Any]] | None = None
        _db: ldb.DBConnection = field(init=False, repr=False)
        _table: ldb.Table | None = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            if self.index_type in ("dense", "hybrid") and self.conversion_func is None:
                raise ValueError(
                    f"conversion_func is required for index_type={self.index_type!r}"
                )

            self._db = ldb.connect(self.uri)

            if self.table in self._db.list_tables().tables:
                self._table = self._db.open_table(self.table)

        def has_index(self) -> bool:
            """Return whether a table exists in the database."""
            return self._table is not None

        def search_limit(self) -> int | None:
            """Return the total number of rows, or ``None`` when empty."""
            if self._table is None:
                return None

            return self._table.count_rows()

        def _build_rows(self, casebase: Casebase[K, str]) -> list[dict[str, Any]]:
            """Build row dicts for LanceDB from a casebase."""
            keys = list(casebase.keys())
            values = list(casebase.values())

            if self.index_type == "sparse":
                rows = [
                    {self.key_column: key, self.value_column: value}
                    for key, value in zip(keys, values, strict=True)
                ]
            else:
                assert self.conversion_func is not None
                vecs = self.conversion_func(values)
                rows = [
                    {
                        self.key_column: key,
                        self.value_column: value,
                        self.vector_column: np.asarray(vec).tolist(),
                    }
                    for key, value, vec in zip(keys, values, vecs, strict=True)
                ]

            if self.metadata_func is not None:
                for row, key, value in zip(rows, keys, values, strict=True):
                    row.update(self.metadata_func(key, value))

            return rows

        def _setup_indices(self, table: ldb.Table) -> None:
            """Create scalar and optional FTS indices on a table."""
            table.create_scalar_index(self.key_column, replace=True)

            if self.index_type in ("sparse", "hybrid"):
                table.create_fts_index(self.value_column, replace=True)

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase from the LanceDB table."""
            if self._table is None:
                return {}
            table = self._table.to_arrow()
            keys = table.column(self.key_column).to_pylist()
            values = table.column(self.value_column).to_pylist()
            return dict(zip(keys, values, strict=True))

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild LanceDB table, reusing existing rows where possible."""
            if not data or self._table is None:
                rows = self._build_rows(data)
                table = self._db.create_table(self.table, rows, mode="overwrite")
                self._setup_indices(table)
                self._table = table
                return

            existing = self.index
            stale_keys, changed_or_new = _compute_index_diff(existing, data)

            if not stale_keys and not changed_or_new:
                return

            keys_to_delete = stale_keys | (set(changed_or_new.keys()) & set(existing.keys()))
            if keys_to_delete:
                self.delete_index(keys_to_delete)

            if changed_or_new:
                rows = self._build_rows(changed_or_new)
                self._table.add(rows)

            self._setup_indices(self._table)

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Append rows to an existing LanceDB table."""
            if self._table is None:
                self.create_index(data)
                return

            rows = self._build_rows(data)
            self._table.add(rows)
            self._setup_indices(self._table)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Delete rows from the LanceDB table by key."""
            if self._table is None:
                return

            if not data:
                return

            key_list = list(data)
            sample = key_list[0]
            col = self.key_column

            if isinstance(sample, str):
                predicate = f"{col} IN (" + ", ".join(f"'{k}'" for k in key_list) + ")"
            else:
                predicate = f"{col} IN (" + ", ".join(str(k) for k in key_list) + ")"

            self._table.delete(predicate)
            self._setup_indices(self._table)


with optional_dependencies():
    import chromadb as cdb
    from chromadb.api import ClientAPI

    @dataclass(slots=True)
    class chromadb[K: str](
        IndexableFunc[Casebase[K, str], Collection[K]]
    ):
        """ChromaDB storage backend.

        Manages a persistent ChromaDB collection.  Supports dense,
        sparse, and hybrid index types which control what embedding
        functions and schema are configured.

        Args:
            path: Directory for PersistentClient storage.
            collection: Collection name.
            index_type: Determines what embeddings and indices are
                configured.  ``"dense"`` uses the embedding function,
                ``"sparse"`` uses the sparse embedding function,
                ``"hybrid"`` uses both.
            embedding_func: ChromaDB ``EmbeddingFunction`` for dense
                embeddings.  Required for ``"dense"`` and ``"hybrid"``.
            sparse_embedding_func: ChromaDB
                ``SparseEmbeddingFunction`` for sparse embeddings.
                Required for ``"sparse"`` and ``"hybrid"``.
            metadata_func: Produces extra metadata per document from
                ``(key, value)``.
            sparse_key: Key name for the sparse vector index in the
                ChromaDB schema.
        """

        path: str
        collection: str
        index_type: Literal["dense", "sparse", "hybrid"] = "dense"
        embedding_func: cdb.EmbeddingFunction[Any] | None = None
        sparse_embedding_func: cdb.SparseEmbeddingFunction[Any] | None = None
        metadata_func: Callable[[K, str], cdb.Metadata] | None = None
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
                    self.collection,
                    embedding_function=self.embedding_func,
                )
            except Exception:
                self._collection = None

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
        ) -> tuple[list[str], list[str], list[cdb.Metadata] | None]:
            """Prepare IDs, documents, and metadatas from a casebase."""
            ids = [str(k) for k in data.keys()]
            values = list(data.values())
            metadatas: list[cdb.Metadata] | None = None

            if self.metadata_func is not None:
                metadatas = [
                    self.metadata_func(k, v)
                    for k, v in zip(data.keys(), values, strict=True)
                ]

            return ids, values, metadatas

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
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild ChromaDB collection, reusing existing documents where possible."""
            if self._collection is None:
                collection = self._client.create_collection(
                    name=self.collection,
                    schema=self._build_schema(),
                    embedding_function=self.embedding_func,
                )

                if data:
                    ids, documents, metadatas = self._prepare_documents(data)
                    collection.add(ids=ids, documents=documents, metadatas=metadatas)

                self._collection = collection
                return

            existing = self.index
            stale_keys, changed_or_new = _compute_index_diff(existing, data)

            if not stale_keys and not changed_or_new:
                return

            if stale_keys:
                self.delete_index(stale_keys)

            if changed_or_new:
                ids, documents, metadatas = self._prepare_documents(changed_or_new)
                self._collection.upsert(
                    ids=ids, documents=documents, metadatas=metadatas
                )

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Upsert documents into an existing ChromaDB collection."""
            if self._collection is None:
                self.create_index(data)
                return

            ids, documents, metadatas = self._prepare_documents(data)
            self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Remove documents by ID from the ChromaDB collection."""
            if self._collection is None:
                return

            ids = [str(k) for k in data]

            if ids:
                self._collection.delete(ids=ids)


with optional_dependencies():
    import numpy as np
    import zvec as zv

    @dataclass(slots=True, frozen=True)
    class _ZvecCasebaseView[K: str](Mapping[K, str]):
        """Lazy mapping backed by a zvec collection.

        Keys are tracked in-memory; values are fetched on demand
        via ``Collection.fetch()``.
        """

        _collection: zv.Collection
        _value_field: str
        _keys: frozenset[K]

        def __getitem__(self, key: K) -> str:
            if key not in self._keys:
                raise KeyError(key)
            result = self._collection.fetch(key)
            if key not in result:
                raise KeyError(key)
            return result[key].field(self._value_field) or ""

        def __iter__(self) -> Iterator[K]:
            return iter(self._keys)

        def __len__(self) -> int:
            return len(self._keys)

        def __hash__(self) -> int:
            return id(self)

    @dataclass(slots=True)
    class zvec[K: str](
        IndexableFunc[Casebase[K, str], Collection[K]]
    ):
        """Zvec storage backend.

        Manages an embedded zvec collection on disk.  Supports dense
        (vector), sparse (sparse vector), and hybrid index types which
        control what data is stored and what indices are built.

        Args:
            path: Directory path for the zvec collection.
            collection: Collection name used in the schema.
            index_type: Determines what vectors are stored and which
                indices are created.  ``"dense"`` stores dense
                embeddings, ``"sparse"`` stores sparse embeddings,
                ``"hybrid"`` stores both.
            conversion_func: Dense embedding function.  Required for
                ``"dense"`` and ``"hybrid"`` index types.
            sparse_conversion_func: Sparse embedding function returning
                ``dict[int, float]`` per document.  Required for
                ``"sparse"`` and ``"hybrid"`` index types.
            metric_type: Distance metric for dense vector search.
            metadata_func: Optional callable that produces extra scalar
                fields for each document.  Called with ``(key, value)``
                and must return a dict mapping field names to values.
                All documents must produce the same set of field names.
            value_field: Field name for storing case text values.
            dense_vector_name: Name for the dense vector field.
            sparse_vector_name: Name for the sparse vector field.
        """

        path: str
        collection: str
        index_type: Literal["dense", "sparse", "hybrid"] = "dense"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        sparse_conversion_func: BatchConversionFunc[str, dict[int, float]] | None = None
        metric_type: Literal["cosine", "ip", "l2"] = "cosine"
        metadata_func: Callable[[K, str], dict[str, Any]] | None = None
        value_field: str = "value"
        dense_vector_name: str = "dense"
        sparse_vector_name: str = "sparse"
        _collection: zv.Collection | None = field(default=None, init=False, repr=False)
        _keys: set[K] | None = field(default=None, init=False, repr=False)
        _metadata_field_names: frozenset[str] | None = field(default=None, init=False, repr=False)

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

        def has_index(self) -> bool:
            """Return whether a collection exists on disk."""
            return self._collection is not None

        def search_limit(self) -> int | None:
            """Return the total number of indexed documents, or ``None`` when empty."""
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

        def _build_schema(self, data: Casebase[K, str]) -> zv.CollectionSchema:
            """Build a CollectionSchema, inferring vector dimension from data."""
            if not data and self.index_type in ("dense", "hybrid"):
                raise ValueError(
                    "Cannot build dense/hybrid schema without data to infer dimension."
                )

            fields: list[zv.FieldSchema] = [
                zv.FieldSchema(self.value_field, zv.DataType.STRING),
            ]
            vectors: list[zv.VectorSchema] = []

            if self.index_type in ("dense", "hybrid"):
                assert self.conversion_func is not None
                sample = self.conversion_func([next(iter(data.values()))])
                dimension = len(np.asarray(sample[0]))
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

            if self.metadata_func is not None and data:
                sample_key = next(iter(data.keys()))
                sample_meta = self.metadata_func(sample_key, data[sample_key])
                for fname, fval in sample_meta.items():
                    fields.append(
                        zv.FieldSchema(fname, self._infer_field_type(fval))
                    )

            return zv.CollectionSchema(
                self.collection, fields=fields, vectors=vectors
            )

        def _build_docs(self, casebase: Casebase[K, str]) -> list[zv.Doc]:
            """Build zvec Doc objects from a casebase."""
            keys = list(casebase.keys())
            values = list(casebase.values())

            dense_vecs = None
            sparse_vecs = None

            if self.index_type in ("dense", "hybrid"):
                assert self.conversion_func is not None
                dense_vecs = self.conversion_func(values)

            if self.index_type in ("sparse", "hybrid"):
                assert self.sparse_conversion_func is not None
                sparse_vecs = self.sparse_conversion_func(values)

            docs: list[zv.Doc] = []

            for i, (key, value) in enumerate(zip(keys, values, strict=True)):
                doc_vectors: dict[str, Any] = {}
                doc_fields: dict[str, Any] = {self.value_field: value}

                if dense_vecs is not None:
                    doc_vectors[self.dense_vector_name] = (
                        np.asarray(dense_vecs[i]).tolist()
                    )

                if sparse_vecs is not None:
                    doc_vectors[self.sparse_vector_name] = sparse_vecs[i]

                if self.metadata_func is not None:
                    meta = self.metadata_func(key, value)
                    if self._metadata_field_names is None:
                        self._metadata_field_names = frozenset(meta.keys())
                    elif set(meta.keys()) != self._metadata_field_names:
                        raise ValueError(
                            f"metadata_func returned fields {set(meta.keys())} for key={key!r}, "
                            f"expected {self._metadata_field_names}"
                        )
                    doc_fields.update(meta)

                docs.append(
                    zv.Doc(id=str(key), vectors=doc_vectors, fields=doc_fields)
                )

            return docs

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase."""
            if self._keys is None or self._collection is None:
                return {}
            return _ZvecCasebaseView(self._collection, self.value_field, frozenset(self._keys))

        @override
        def create_index(self, data: Casebase[K, str]) -> None:
            """Rebuild zvec collection, reusing existing documents where possible."""
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

            if stale_keys:
                self.delete_index(stale_keys)

            if changed_or_new:
                docs = self._build_docs(changed_or_new)
                self._collection.upsert(docs)

            self._keys = set(data.keys())

        @override
        def update_index(self, data: Casebase[K, str]) -> None:
            """Upsert documents into an existing zvec collection."""
            if self._collection is None:
                self.create_index(data)
                return

            docs = self._build_docs(data)
            self._collection.upsert(docs)

            if self._keys is None:
                self._keys = set(data.keys())
            else:
                self._keys.update(data.keys())

        @override
        def delete_index(self, data: Collection[K]) -> None:
            """Remove documents by ID from the zvec collection."""
            if self._collection is None:
                return

            ids = [str(k) for k in data]

            if ids:
                self._collection.delete(ids)

            if self._keys is not None:
                self._keys -= set(data)
