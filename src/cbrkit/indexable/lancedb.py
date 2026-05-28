"""LanceDB storage backend."""

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any, Literal, cast, override

import lancedb as ldb
import numpy as np

from ..helpers import get_logger
from ..typing import BatchConversionFunc, Casebase, IndexableFunc, NumpyArray
from ._common import _normalize_patch_keys, _sql_in_clause

logger = get_logger(__name__)


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


__all__ = ["lancedb"]
