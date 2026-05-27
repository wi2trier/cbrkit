"""Pure storage backends for indexable retrieval.

This module provides storage classes that implement
:class:`~cbrkit.typing.IndexableFunc` without any retrieval logic.
Each backend manages database connections, data ingestion, and index
maintenance.  Retriever wrappers in :mod:`cbrkit.retrieval` consume
these storage instances to perform search queries.

Example:
    >>> import tempfile  # doctest: +SKIP
    >>> storage = lancedb(  # doctest: +SKIP
    ...     uri=tempfile.mkdtemp(),
    ...     table_name="cases",
    ...     index_type="sparse",
    ... )
    >>> storage.put_index({0: "hello world", 1: "foo bar"})  # doctest: +SKIP
    >>> storage.has_index()  # doctest: +SKIP
    True
"""

from collections.abc import Callable, Collection, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

from .helpers import dist2sim, get_logger, optional_dependencies
from .typing import (
    BatchConversionFunc,
    Casebase,
    IndexableFunc,
    NumpyArray,
    SparseVector,
)

__all__ = [
    "chromadb",
    "lancedb",
    "pgvector",
    "zvec",
]

logger = get_logger(__name__)


def _compute_index_diff[K](
    existing: Casebase[K, str],
    data: Casebase[K, str],
) -> tuple[set[K], Casebase[K, str]]:
    """Compute stale keys and changed/new entries for index updates.

    Args:
        existing: The current indexed casebase.
        data: The desired casebase to sync with.

    Returns:
        A tuple `(stale_keys, changed_or_new)` where *stale_keys* are
        keys present in *existing* but absent from *data*, and
        *changed_or_new* maps keys whose values differ or are new.
    """
    new_keys = set(data.keys())
    old_keys = set(existing.keys())
    stale_keys = old_keys - new_keys
    changed_or_new: Casebase[K, str] = {
        k: data[k] for k in new_keys if k not in existing or existing[k] != data[k]
    }
    return stale_keys, changed_or_new


def _normalize_patch_keys[K](
    upsert: Collection[K] | None,
    delete: Collection[K] | None,
) -> tuple[set[K], set[K]] | None:
    """Validate `patch_index` arguments and split into key sets.

    Mappings count as collections of their keys, so this helper works
    for both keyed (`Casebase[K, str]`) and bare (`Collection[str]`)
    patch arguments.

    Returns `None` when both inputs are empty (the patch is a no-op).
    Raises `ValueError` if the same key appears in both *upsert* and
    *delete*.
    """
    if not upsert and not delete:
        return None

    upsert_keys = set(upsert) if upsert else set()
    delete_keys = set(delete) if delete else set()
    overlap = upsert_keys & delete_keys

    if overlap:
        raise ValueError(f"Cannot upsert and delete the same entries: {overlap!r}")

    return upsert_keys, delete_keys


def _sql_literal(value: int | str) -> str:
    """Return a SQL literal for supported index keys."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"

    return str(value)


def _sql_identifier(name: str) -> str:
    """Quote *name* as a double-quoted SQL identifier with embedded `\"` escaped."""
    return '"' + name.replace('"', '""') + '"'


def _sql_in_clause[K: int | str](column: str, keys: Collection[K]) -> str:
    """Build a SQL `IN (...)` predicate for supported index keys."""
    return f"{column} IN ({', '.join(_sql_literal(k) for k in keys)})"


with optional_dependencies():
    import lancedb as ldb
    import numpy as np

    @dataclass(slots=True)
    class lancedb[K: int | str](IndexableFunc[Casebase[K, str], Collection[K]]):
        """LanceDB storage backend.

        Manages an embedded LanceDB database on disk.  Supports dense
        (vector), sparse (FTS/BM25), and hybrid index types which
        control what data is stored and what indices are built.

        Warning:
            Persisted vectors are tied to the
            :paramref:`conversion_func` used when they were written.
            Reopening a table backed by a different embedding model
            silently returns wrong results when the new model has the
            same dimension, and raises on `INSERT` when it does not —
            :meth:`put_index` only re-embeds entries whose text
            changed.  Drop the table (or use a fresh `table_name`)
            when changing models.

        Args:
            uri: Path to the LanceDB database directory.
            table_name: Table name within the database.
            index_type: Determines what data is stored and which
                indices are created.  `"dense"` stores embeddings,
                `"sparse"` builds an FTS index, `"hybrid"` does
                both.
            conversion_func: Embedding function.  Required for
                `"dense"` and `"hybrid"` index types.
            key_column: Column name for case keys.
            value_column: Column name for case text values.
            vector_column: Column name for dense embedding vectors.
            metadata_func: Optional callable that produces extra
                columns for each row.  Called with `(key, value)`
                and must return a dict mapping column names to values.
        """

        uri: str
        table_name: str
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

            if self.table_name in self._db.list_tables().tables:
                self._table = self._db.open_table(self.table_name)

        @override
        def has_index(self) -> bool:
            """Return whether a table exists in the database."""
            return self._table is not None

        def search_limit(self) -> int | None:
            """Return the total number of rows, or `None` when empty."""
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
        def put_index(
            self,
            data: Casebase[K, str],
        ) -> None:
            """Replace the LanceDB table contents with *data*."""
            if self._table is None:
                if not data:
                    return

                rows = self._build_rows(data)
                self._table = self._db.create_table(
                    self.table_name,
                    rows,
                    mode="overwrite",
                )
                self._setup_indices(self._table)
                return

            if not data:
                self._table.delete("true")
                return

            rows = self._build_rows(data)
            (
                self._table.merge_insert(self.key_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete()
                .execute(rows)
            )

        @override
        def upsert_index(
            self,
            data: Casebase[K, str],
        ) -> None:
            """Insert or replace rows in the LanceDB table.

            If no table exists yet, delegates to :meth:`put_index`.
            """
            if self._table is None:
                self.put_index(data)
                return

            if not data:
                return

            rows = self._build_rows(data)
            (
                self._table.merge_insert(self.key_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )

        @override
        def delete_index(
            self,
            data: Collection[K],
        ) -> None:
            """Delete rows from the LanceDB table by key."""
            if self._table is None or not data:
                return

            self._table.delete(_sql_in_clause(self.key_column, data))

        @override
        def patch_index(
            self,
            upsert: Casebase[K, str] | None = None,
            delete: Collection[K] | None = None,
        ) -> None:
            """Apply inserts, replacements, and deletes as one LanceDB mutation."""
            normalized = _normalize_patch_keys(upsert, delete)

            if normalized is None:
                return

            _, delete_keys = normalized

            if self._table is None:
                if upsert:
                    self.put_index(upsert)
                return

            if not upsert:
                self.delete_index(delete_keys)
                return

            rows = self._build_rows(upsert)
            operation = (
                self._table.merge_insert(self.key_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
            )

            if delete_keys:
                operation = operation.when_not_matched_by_source_delete(
                    _sql_in_clause(self.key_column, delete_keys)
                )

            operation.execute(rows)

        def keys_where(self, where: str | None = None) -> list[K]:
            """Return keys matching a native LanceDB predicate."""
            if self._table is None:
                return []

            query = self._table.search().select([self.key_column])

            if where is not None:
                query = query.where(where)

            table = query.to_arrow()
            return cast(list[K], table.column(self.key_column).to_pylist())

        def delete_where(
            self,
            where: str,
        ) -> list[K]:
            """Delete rows matching a native LanceDB predicate and return their keys."""
            if self._table is None:
                return []

            keys = self.keys_where(where)

            if not keys:
                return []

            self._table.delete(where)
            return keys

        def replace_where(self, where: str, data: Casebase[K, str]) -> list[K]:
            """Replace rows matching a native LanceDB predicate with *data*."""
            if self._table is None:
                self.put_index(data)
                return []

            keys = self.keys_where(where)

            if not data:
                if keys:
                    self._table.delete(where)
                return keys

            rows = self._build_rows(data)
            (
                self._table.merge_insert(self.key_column)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .when_not_matched_by_source_delete(where)
                .execute(rows)
            )
            return keys


with optional_dependencies():
    import chromadb as cdb
    from chromadb.api import ClientAPI

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
            metadata_func: Produces extra metadata per document from
                `(key, value)`.
            sparse_key: Key name for the sparse vector index in the
                ChromaDB schema.
        """

        path: str
        collection_name: str
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
                    self.collection_name,
                    embedding_function=self.embedding_func,
                )
            except Exception:
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
        def upsert_index(
            self,
            data: Casebase[K, str],
        ) -> None:
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
            upsert: Casebase[K, str] | None = None,
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


with optional_dependencies():
    import numpy as np
    import zvec as zv  # pyright: ignore[reportMissingImports]  # type: ignore[unresolved-import]

    @dataclass(slots=True, frozen=True)
    class _ZvecCasebaseView[K: str](Mapping[K, str]):
        """Lazy mapping backed by a zvec collection.

        Keys are tracked in-memory; values are fetched on demand
        via `Collection.fetch()`.
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
            value = result[key].field(self._value_field)
            return cast(str, value) if value else ""

        def __iter__(self) -> Iterator[K]:
            return iter(self._keys)

        def __len__(self) -> int:
            return len(self._keys)

        def __hash__(self) -> int:
            return id(self)

    @dataclass(slots=True)
    class zvec[K: str](IndexableFunc[Casebase[K, str], Collection[K]]):
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
            metadata_func: Optional callable that produces extra scalar
                fields for each document.  Called with `(key, value)`
                and must return a dict mapping field names to values.
                All documents must produce the same set of field names.
            value_field: Field name for storing case text values.
            dense_vector_name: Name for the dense vector field.
            sparse_vector_name: Name for the sparse vector field.
        """

        path: str
        collection_name: str
        index_type: Literal["dense", "sparse", "hybrid"] = "dense"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        sparse_conversion_func: BatchConversionFunc[str, SparseVector] | None = None
        metric_type: Literal["cosine", "ip", "l2"] = "cosine"
        metadata_func: Callable[[K, str], dict[str, Any]] | None = None
        value_field: str = "value"
        dense_vector_name: str = "dense"
        sparse_vector_name: str = "sparse"
        _collection: zv.Collection | None = field(default=None, init=False, repr=False)
        _keys: set[K] | None = field(default=None, init=False, repr=False)
        _metadata_field_names: frozenset[str] | None = field(
            default=None, init=False, repr=False
        )

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
                    fields.append(zv.FieldSchema(fname, self._infer_field_type(fval)))

            return zv.CollectionSchema(
                self.collection_name, fields=fields, vectors=vectors
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
                    doc_vectors[self.dense_vector_name] = np.asarray(
                        dense_vecs[i]
                    ).tolist()

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

                docs.append(zv.Doc(id=str(key), vectors=doc_vectors, fields=doc_fields))

            return docs

        @property
        @override
        def index(self) -> Casebase[K, str]:
            """Return the indexed casebase."""
            if self._keys is None or self._collection is None:
                return {}
            return _ZvecCasebaseView(
                self._collection, self.value_field, frozenset(self._keys)
            )

        @override
        def put_index(
            self,
            data: Casebase[K, str],
        ) -> None:
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
        def upsert_index(
            self,
            data: Casebase[K, str],
        ) -> None:
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
            upsert: Casebase[K, str] | None = None,
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


@dataclass(slots=True, frozen=True)
class _PgMetric:
    """Per-metric pgvector configuration."""

    opclass: str
    """HNSW operator class name passed to `CREATE INDEX ... USING hnsw`."""
    distance_attr: str
    """`pgvector.sqlalchemy.Vector` method name returning the distance expression."""
    sim_from_distance: Callable[[float], float]
    """Convert a raw pgvector distance to a similarity score."""


PG_METRICS: dict[Literal["cosine", "ip", "l2"], _PgMetric] = {
    "cosine": _PgMetric("vector_cosine_ops", "cosine_distance", dist2sim),
    "ip": _PgMetric("vector_ip_ops", "max_inner_product", lambda d: -d),
    "l2": _PgMetric("vector_l2_ops", "l2_distance", dist2sim),
}


with optional_dependencies():
    import numpy as np
    import sqlalchemy as sa
    from pgvector.sqlalchemy import Vector
    from sqlalchemy.dialects.postgresql import TSVECTOR, insert as pg_insert

    @dataclass(slots=True)
    class pgvector[K: int | str](IndexableFunc[Casebase[K, str], Collection[K]]):
        """PostgreSQL/pgvector storage backend.

        Manages a single table in a PostgreSQL database, using
        :mod:`pgvector` for dense vectors and built-in `tsvector`
        full-text search for the sparse path.  Supports `"dense"`,
        `"sparse"`, and `"hybrid"` index types.

        The table is created lazily on the first :meth:`put_index` call.
        The vector dimension is inferred from the first batch of
        embeddings, and the `vector` extension is created automatically
        for dense/hybrid indexes if missing.  Pre-existing tables are
        opened by reflection without any schema validation — mismatched
        column names or types surface as `INSERT`/`SELECT` errors from
        PostgreSQL.

        Warning:
            Persisted vectors are tied to the
            :paramref:`conversion_func` used when they were written.
            Reopening a table backed by a different embedding model
            silently returns wrong results when the new model has the
            same dimension, and raises on `INSERT` when it does not.
            Drop the table (or use a fresh `table_name`) when changing
            models.

        Args:
            url: SQLAlchemy database URL
                (e.g. ``postgresql+psycopg://user:pw@host/db`` for
                psycopg v3, ``postgresql+psycopg2://...`` for
                psycopg2).  This extra deliberately does not pull in
                a PostgreSQL DBAPI driver — install one matching the
                URL scheme separately
                (``pip install psycopg[binary]`` is the modern choice).
            table_name: Table name within the database.
            index_type: Determines what columns and indices are
                created.  `"dense"` stores embeddings, `"sparse"`
                stores a `tsvector`, `"hybrid"` does both.
            conversion_func: Embedding function.  Required for
                `"dense"` and `"hybrid"` index types.
            key_type: Column type for case keys (`"int"` or `"str"`).
                Must match the generic parameter `K`.
            text_search_config: PostgreSQL FTS configuration name used
                for `to_tsvector` and `plainto_tsquery`.
            metric_type: Distance metric used to build the HNSW index
                on the vector column.
            key_column: Column name for case keys.
            value_column: Column name for case text values.
            vector_column: Column name for dense embeddings.
            tsv_column: Column name for the `tsvector` FTS index.
            metadata_func: Optional callable returning extra columns
                per row from `(key, value)`.  All rows must produce
                the same keys; column types are inferred from the
                first row at table creation time.
        """

        url: str
        table_name: str
        index_type: Literal["dense", "sparse", "hybrid"] = "dense"
        conversion_func: BatchConversionFunc[str, NumpyArray] | None = None
        key_type: Literal["int", "str"] = "str"
        text_search_config: str = "english"
        metric_type: Literal["cosine", "ip", "l2"] = "cosine"
        key_column: str = "key"
        value_column: str = "value"
        vector_column: str = "vector"
        tsv_column: str = "tsv"
        metadata_func: Callable[[K, str], dict[str, Any]] | None = None
        _engine: sa.Engine = field(init=False, repr=False)
        _metadata: sa.MetaData = field(init=False, repr=False)
        _table: sa.Table | None = field(default=None, init=False, repr=False)

        def __post_init__(self) -> None:
            if self.index_type in ("dense", "hybrid") and self.conversion_func is None:
                raise ValueError(
                    f"conversion_func is required for index_type={self.index_type!r}"
                )

            self._engine = sa.create_engine(self.url)
            self._metadata = sa.MetaData()

            with self._engine.connect() as conn:
                if sa.inspect(conn).has_table(self.table_name):
                    self._table = sa.Table(
                        self.table_name, self._metadata, autoload_with=conn
                    )

        def close(self) -> None:
            """Dispose of the SQLAlchemy engine and its connection pool."""
            self._engine.dispose()

        def _cast_key(self, value: Any) -> K:
            return cast(K, int(value) if self.key_type == "int" else str(value))

        @staticmethod
        def _infer_sa_type(value: Any) -> sa.types.TypeEngine[Any]:
            if isinstance(value, bool):
                return sa.Boolean()
            if isinstance(value, int):
                return sa.BigInteger()
            if isinstance(value, float):
                return sa.Float()
            return sa.Text()

        def _build_rows(self, data: Casebase[K, str]) -> list[dict[str, Any]]:
            """Build row dicts with vectors and metadata from a casebase."""
            keys = list(data.keys())
            values = list(data.values())

            rows: list[dict[str, Any]] = [
                {self.key_column: k, self.value_column: v}
                for k, v in zip(keys, values, strict=True)
            ]

            if self.index_type in ("dense", "hybrid"):
                assert self.conversion_func is not None
                for row, vec in zip(rows, self.conversion_func(values), strict=True):
                    row[self.vector_column] = np.asarray(vec).tolist()

            if self.metadata_func is not None:
                for row, k, v in zip(rows, keys, values, strict=True):
                    row.update(self.metadata_func(k, v))

            return rows

        def _build_table(self, sample: dict[str, Any]) -> sa.Table:
            """Build a `sa.Table` definition by inspecting the first row."""
            key_sa = sa.BigInteger() if self.key_type == "int" else sa.Text()
            columns: list[sa.Column[Any]] = [
                sa.Column(self.key_column, key_sa, primary_key=True),
                sa.Column(self.value_column, sa.Text(), nullable=False),
            ]

            if self.index_type in ("dense", "hybrid"):
                columns.append(
                    sa.Column(
                        self.vector_column,
                        Vector(len(sample[self.vector_column])),
                        nullable=False,
                    )
                )

            if self.index_type in ("sparse", "hybrid"):
                tsv_expr = (
                    f"to_tsvector({_sql_literal(self.text_search_config)}, "
                    f"{_sql_identifier(self.value_column)})"
                )
                columns.append(
                    sa.Column(
                        self.tsv_column,
                        TSVECTOR(),
                        sa.Computed(tsv_expr, persisted=True),
                        nullable=False,
                    )
                )

            reserved = {c.name for c in columns}
            for fname, fval in sample.items():
                if fname not in reserved:
                    columns.append(
                        sa.Column(fname, self._infer_sa_type(fval), nullable=True)
                    )

            return sa.Table(self.table_name, self._metadata, *columns)

        def _create_indices(self, conn: sa.Connection, table: sa.Table) -> None:
            if self.index_type in ("dense", "hybrid"):
                sa.Index(
                    f"ix_{self.table_name}_{self.vector_column}",
                    table.c[self.vector_column],
                    postgresql_using="hnsw",
                    postgresql_ops={
                        self.vector_column: PG_METRICS[self.metric_type].opclass
                    },
                ).create(conn, checkfirst=True)

            if self.index_type in ("sparse", "hybrid"):
                sa.Index(
                    f"ix_{self.table_name}_{self.tsv_column}",
                    table.c[self.tsv_column],
                    postgresql_using="gin",
                ).create(conn, checkfirst=True)

        def _upsert_stmt(self, table: sa.Table, rows: list[dict[str, Any]]) -> Any:
            """Build an `INSERT ... ON CONFLICT DO UPDATE` statement."""
            stmt = pg_insert(table).values(rows)
            update_cols = [c for c in rows[0] if c != self.key_column]
            return stmt.on_conflict_do_update(
                index_elements=[table.c[self.key_column]],
                set_={c: stmt.excluded[c] for c in update_cols},
            )

        @override
        def has_index(self) -> bool:
            return self._table is not None

        def search_limit(self) -> int | None:
            """Return the total number of rows, or `None` when no table exists."""
            if self._table is None:
                return None
            with self._engine.connect() as conn:
                return int(
                    conn.execute(
                        sa.select(sa.func.count()).select_from(self._table)
                    ).scalar_one()
                )

        @property
        @override
        def index(self) -> Casebase[K, str]:
            if self._table is None:
                return {}
            kc = self._table.c[self.key_column]
            vc = self._table.c[self.value_column]
            with self._engine.connect() as conn:
                rows = conn.execute(sa.select(kc, vc)).all()
            return {self._cast_key(k): v for k, v in rows}

        @override
        def put_index(self, data: Casebase[K, str]) -> None:
            """Replace the table contents with *data*.

            On first call the table is created from scratch and bulk-
            loaded before its HNSW/GIN indices are built (faster than
            building indices first).  On subsequent calls only stale or
            text-changed entries are re-embedded; metadata is refreshed
            together with the text.  Empty *data* clears the table.
            """
            if self._table is None:
                if not data:
                    return

                rows = self._build_rows(data)
                table = self._build_table(rows[0])

                with self._engine.begin() as conn:
                    if self.index_type in ("dense", "hybrid"):
                        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
                    table.create(conn)
                    conn.execute(sa.insert(table), rows)
                    self._create_indices(conn, table)

                self._table = table
                return

            if not data:
                with self._engine.begin() as conn:
                    conn.execute(sa.delete(self._table))
                return

            stale_keys, changed = _compute_index_diff(self.index, data)
            self.patch_index(upsert=changed or None, delete=stale_keys or None)

        @override
        def upsert_index(self, data: Casebase[K, str]) -> None:
            if self._table is None:
                self.put_index(data)
                return
            if not data:
                return
            with self._engine.begin() as conn:
                conn.execute(self._upsert_stmt(self._table, self._build_rows(data)))

        @override
        def delete_index(self, data: Collection[K]) -> None:
            if self._table is None or not data:
                return
            with self._engine.begin() as conn:
                conn.execute(
                    sa.delete(self._table).where(
                        self._table.c[self.key_column].in_(list(data))
                    )
                )

        @override
        def patch_index(
            self,
            upsert: Casebase[K, str] | None = None,
            delete: Collection[K] | None = None,
        ) -> None:
            normalized = _normalize_patch_keys(upsert, delete)
            if normalized is None:
                return
            _, delete_keys = normalized

            if self._table is None:
                if upsert:
                    self.put_index(upsert)
                return

            rows = self._build_rows(upsert) if upsert else None
            with self._engine.begin() as conn:
                if delete_keys:
                    conn.execute(
                        sa.delete(self._table).where(
                            self._table.c[self.key_column].in_(list(delete_keys))
                        )
                    )
                if rows:
                    conn.execute(self._upsert_stmt(self._table, rows))
