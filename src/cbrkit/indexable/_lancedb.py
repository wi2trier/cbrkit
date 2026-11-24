"""LanceDB storage backend."""

import asyncio
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Literal, cast, override

import lancedb as ldb
import numpy as np
from lancedb.index import FTS, BTree
from lancedb.query import AsyncStandardQuery
from lancedb.table import AsyncTable

from .._util import compute_index_diff, identity
from ..helpers import dist2sim
from ..sim.embed.metrics import Cosine, Dot, Euclidean
from ..typing import (
    Casebase,
    EmbeddingFunc,
    FilterableIndex,
    NumpyArray,
    Results,
    SimMap,
)
from ._common import RowCodec, VectorMetric, make_codec
from ._retriever import (
    IndexType,
    VectorStorageRetriever,
    check_embedder,
    dense_embedder,
    embed_rows,
    has_sparse,
)


def _sql_literal(value: int | str) -> str:
    """Return a SQL literal for supported index keys."""
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"

    return str(value)


def _sql_in_clause(column: str, keys: Collection[int | str]) -> str:
    """Build a SQL `IN (...)` predicate for supported index keys."""
    return f"{column} IN ({', '.join(_sql_literal(k) for k in keys)})"


_METRICS: Mapping[
    type[VectorMetric], tuple[Literal["cosine", "dot", "l2"], Callable[[float], float]]
] = {
    Cosine: ("cosine", dist2sim),
    # LanceDB's dot distance is one minus the dot product
    Dot: ("dot", lambda distance: 1 - distance),
    Euclidean: ("l2", dist2sim),
}


@dataclass(slots=True, frozen=True)
class LanceDB[K: int | str, V = str](FilterableIndex[K, V, str]):
    """LanceDB storage backend.

    Stores the cases in a table of a LanceDB database that the host opens with `lancedb.connect_async`.
    Supports dense (vector), sparse (FTS/BM25), and hybrid index types,
    which control what data is stored and what indices are built.
    Every operation opens the table, so the storage does no I/O until it is used
    and the table is created by the first write.
    `put_index` writes the difference to the stored cases,
    and `replace_where` and `delete_where` change the cases matching a predicate.

    It is no `cbrkit.typing.Storage`, so it cannot store retained cases:
    LanceDB has no transaction that allocates keys and checks the stored cases
    together with a write, which retaining needs to keep concurrent runs apart.

    Warning:
        Persisted vectors are tied to the `embedder` used when they were written.
        Reopening a table backed by a different embedding model
        silently returns wrong results when the new model has the
        same dimension, and raises on `INSERT` when it does not,
        since `put_index` only re-embeds entries whose text changed.
        Drop the table (or use a fresh `table_name`) when changing models.

    Args:
        connection: Connection of the LanceDB database, opened and closed by the host.
        table_name: Table name within the database.
        index_type: Determines what data is stored and which indices are created.
            `"dense"` stores embeddings, `"sparse"` builds an FTS index, `"hybrid"` does both.
        embedder: Embedding function, required for the `"dense"` and `"hybrid"` index types.
        metric: Distance metric of a dense search.
        key_column: Column name for case keys.
        value_column: Column holding the embeddable text.  With the
            default `V = str` the casebase value is this column,
            with a `model` it names the model field to embed.
        vector_column: Column name for dense embedding vectors.
        model: A dataclass or pydantic `pydantic.BaseModel`
            describing rows richer than plain text.  When set, `V` is
            the model type: every field becomes a stored column,
            `value_column` names the embeddable field, and reads
            reconstruct model instances.

    Examples:
        >>> import asyncio, tempfile
        >>> import lancedb
        >>> async def main() -> list[int]:
        ...     connection = await lancedb.connect_async(tempfile.mkdtemp())
        ...     storage = LanceDB[int, str](connection, index_type="sparse")
        ...     await storage.put_index({0: "hello world", 1: "foo bar"})
        ...     retriever = LanceDBRetriever(storage=storage, limit=1)
        ...     _, sims = (await retriever([({}, "hello")]))[0]
        ...     return list(sims)
        >>> asyncio.run(main())
        [0]
    """

    connection: ldb.AsyncConnection
    _: KW_ONLY
    table_name: str = "cases"
    index_type: IndexType = "dense"
    embedder: EmbeddingFunc[str, NumpyArray] | None = None
    metric: VectorMetric = field(default_factory=Cosine)
    key_column: str = "key"
    value_column: str = "value"
    vector_column: str = "vector"
    model: type[V] | None = None
    _codec: RowCodec[V] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        check_embedder(self.index_type, self.embedder)
        object.__setattr__(
            self,
            "_codec",
            make_codec(self.model, self.value_column, key_column=self.key_column),
        )

    async def table(self) -> AsyncTable | None:
        """Open the table, None if it does not exist yet."""
        try:
            return await self.connection.open_table(self.table_name)
        except ValueError:
            return None

    @override
    async def has_index(self) -> bool:
        """Return whether the table exists."""
        return await self.table() is not None

    async def _build_rows(self, casebase: Casebase[K, V]) -> list[dict[str, Any]]:
        """Build row dicts for LanceDB from a casebase."""
        rows = [self._codec.encode(value) for value in casebase.values()]

        if (embedder := dense_embedder(self.index_type, self.embedder)) is not None:
            await embed_rows(
                embedder,
                rows,
                self.value_column,
                self.vector_column,
                lambda vec: np.asarray(vec).tolist(),
            )

        for row, key in zip(rows, casebase, strict=True):
            row[self.key_column] = key

        return rows

    async def _create(self, data: Casebase[K, V]) -> None:
        """Create the table with `data` and its scalar and optional FTS indices."""
        table = await self.connection.create_table(
            self.table_name, await self._build_rows(data), mode="overwrite"
        )
        await table.create_index(self.key_column, config=BTree())

        if has_sparse(self.index_type):
            await table.create_index(self.value_column, config=FTS())

    async def _merge(
        self, table: AsyncTable, data: Casebase[K, V], delete: str | None
    ) -> None:
        """Upsert `data` and delete the other rows matching `delete` in one merge."""
        operation = (
            table.merge_insert(self.key_column)
            .when_matched_update_all()
            .when_not_matched_insert_all()
        )

        if delete is not None:
            operation = operation.when_not_matched_by_source_delete(delete)

        await operation.execute(await self._build_rows(data))

    @override
    async def get_index(self) -> Casebase[K, V]:
        """Return the cases of the table."""
        return await self._read()

    async def _read(self, where: str | None = None) -> Casebase[K, V]:
        if (table := await self.table()) is None:
            return {}

        columns = self._codec.columns
        query = table.query().select([self.key_column, *columns])

        if where is not None:
            query = query.where(where)

        data = await query.to_arrow()
        payloads = {c: data.column(c).to_pylist() for c in columns}

        return {
            key: self._codec.decode({c: payloads[c][i] for c in columns}, key)
            for i, key in enumerate(data.column(self.key_column).to_pylist())
        }

    @override
    async def put_index(self, data: Casebase[K, V], /) -> None:
        """Replace the cases of the table with `data`.

        Only stale or changed cases are deleted or upserted in one merge, so unchanged cases skip re-embedding.
        """
        if (table := await self.table()) is None:
            if data:
                await self._create(data)

            return

        stale, changed = compute_index_diff(await self._read(), data)
        delete = _sql_in_clause(self.key_column, stale) if stale else None

        if changed:
            await self._merge(table, changed, delete)
        elif delete is not None:
            await table.delete(delete)

    async def _keys_where(self, table: AsyncTable, where: str) -> list[K]:
        data = await table.query().select([self.key_column]).where(where).to_arrow()

        return cast(list[K], data.column(self.key_column).to_pylist())

    @override
    async def keys_where(self, where: str, /) -> list[K]:
        """Return keys matching a native LanceDB predicate."""
        if (table := await self.table()) is None:
            return []

        return await self._keys_where(table, where)

    @override
    async def get_where(self, where: str, /) -> Casebase[K, V]:
        """Return the cases matching a native LanceDB predicate."""
        return await self._read(where)

    @override
    async def delete_where(self, where: str, /) -> list[K]:
        """Delete rows matching a native LanceDB predicate and return their keys."""
        if (table := await self.table()) is None:
            return []

        keys = await self._keys_where(table, where)

        if keys:
            await table.delete(where)

        return keys

    @override
    async def replace_where(self, where: str, data: Casebase[K, V], /) -> list[K]:
        """Replace rows matching a native LanceDB predicate with `data`."""
        if (table := await self.table()) is None:
            await self.put_index(data)
            return []

        keys = await self._keys_where(table, where)

        if data:
            await self._merge(table, data, where)
        elif keys:
            await table.delete(where)

        return keys


@dataclass(slots=True, frozen=True, kw_only=True)
class LanceDBRetriever[K: int | str](VectorStorageRetriever[K, LanceDB[K, Any]]):
    """Search of a `LanceDB` storage.

    Several retrievers can share one storage to search it with different settings.

    Args:
        where: SQL filter expression applied to every search, such as `"category = 'A'"`.
    """

    where: str | None = None

    async def _run[Q: AsyncStandardQuery](
        self,
        queries: Sequence[str],
        build: Callable[[AsyncTable, str, NumpyArray | None], Q],
        score_column: str,
        score: Callable[[float], float],
        embed: bool,
    ) -> Results[K, str]:
        """Run one LanceDB query per query text and fold its hits into results."""
        if (table := await self.storage.table()) is None:
            return [({}, {}) for _ in queries]

        vectors = await self._embed(queries) if embed else [None] * len(queries)
        limit = self.limit if self.limit is not None else await table.count_rows()
        kc, vc = self.storage.key_column, self.storage.value_column

        async def search(
            query: str, vector: NumpyArray | None
        ) -> tuple[Casebase[K, str], SimMap[K, float]]:
            q = build(table, query, vector).limit(limit)

            if self.where is not None:
                q = q.where(self.where)

            hits = await q.to_list()

            return (
                {hit[kc]: hit[vc] for hit in hits},
                {hit[kc]: score(float(hit[score_column])) for hit in hits},
            )

        return await asyncio.gather(
            *(
                search(query, vector)
                for query, vector in zip(queries, vectors, strict=True)
            )
        )

    @override
    async def _search_dense(self, queries: Sequence[str]) -> Results[K, str]:
        distance_type, score = _METRICS[type(self.storage.metric)]

        return await self._run(
            queries,
            lambda table, _, vector: (
                table.vector_search(np.asarray(vector).tolist())
                .column(self.storage.vector_column)
                .distance_type(distance_type)
            ),
            "_distance",
            score,
            embed=True,
        )

    @override
    async def _search_sparse(self, queries: Sequence[str]) -> Results[K, str]:
        return await self._run(
            queries,
            lambda table, query, _: table.query().nearest_to_text(query),
            "_score",
            identity,
            embed=False,
        )


__all__ = [
    "LanceDB",
    "LanceDBRetriever",
]
