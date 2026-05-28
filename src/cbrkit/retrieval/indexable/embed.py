"""Embedding-based retriever with indexing support."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import override

from ...helpers import batchify_sim, dispatch_batches
from ...indexable._common import _compute_index_diff, _normalize_patch_keys
from ...sim.embed import cache, default_score_func
from ...typing import (
    AnySimFunc,
    BatchSimFunc,
    Casebase,
    Float,
    IndexableFunc,
    NumpyArray,
    RetrieverFunc,
)
from ._common import resolve_casebases


@dataclass(slots=True, init=False)
class embed[K, S: Float](
    RetrieverFunc[K, str, S],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """Embedding-based semantic retriever with indexing support.

    Warning:
        The :class:`~cbrkit.sim.embed.cache` instance passed as
        :paramref:`conversion_func` keys vectors by text, not by model
        identity, so reusing a persistent cache across runs with a
        different model returns silently stale vectors for unchanged
        texts.  Drop the cache (or use a fresh `table`) when changing
        models.

    Args:
        conversion_func: Embedding function (from embed module).
        sim_func: Vector similarity function (default: cosine).
        query_conversion_func: Optional separate embedding function for queries.

    Pass an empty casebase to `__call__` to use the pre-indexed casebase.
    """

    conversion_func: cache
    sim_func: BatchSimFunc[NumpyArray, S]
    query_conversion_func: cache | None
    _casebase: dict[K, str] | None = field(repr=False, init=False, default=None)

    def __init__(
        self,
        conversion_func: cache,
        sim_func: AnySimFunc[NumpyArray, S] = default_score_func,  # type: ignore[assignment]  # ty: ignore[invalid-parameter-default]
        query_conversion_func: cache | None = None,
    ):
        self.conversion_func = conversion_func
        self.sim_func = batchify_sim(sim_func)
        self.query_conversion_func = query_conversion_func
        self._casebase = None

    @override
    def has_index(self) -> bool:
        """Return whether an embedding index has been created."""
        return self._casebase is not None

    @property
    @override
    def index(self) -> Casebase[K, str]:
        """Return the indexed casebase."""
        if self._casebase is None:
            return {}
        return self._casebase

    @override
    def put_index(
        self,
        data: Casebase[K, str],
    ) -> None:
        """Replace the embedding index contents with *data*.

        On first call the casebase and embedding cache are built from
        scratch.  On subsequent calls only stale or changed entries
        are removed/added, so unchanged texts skip re-embedding.
        """
        if self._casebase is None:
            self._casebase = dict(data)
            # Additive upsert preserves any vectors already in a shared cache.
            self.conversion_func.upsert_index(data.values())
            return

        existing = self._casebase
        stale_keys, changed_or_new = _compute_index_diff(existing, data)

        if not stale_keys and not changed_or_new:
            return

        self.patch_index(
            upsert=changed_or_new or None,
            delete=stale_keys or None,
        )

    def _obsolete_texts(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> list[str]:
        """Cached texts that become unreferenced after applying *upsert*/*delete*.

        Skips texts still referenced by other unaffected keys so the
        cache does not lose entries shared across multiple keys.
        """
        assert self._casebase is not None
        candidates: set[str] = set()
        affected_keys: set[K] = set()

        if upsert:
            for key, new_value in upsert.items():
                if key in self._casebase and self._casebase[key] != new_value:
                    candidates.add(self._casebase[key])
                    affected_keys.add(key)

        if delete:
            for key in delete:
                if key in self._casebase:
                    candidates.add(self._casebase[key])
                    affected_keys.add(key)

        still_referenced = {
            text for key, text in self._casebase.items() if key not in affected_keys
        }
        return [text for text in candidates if text not in still_referenced]

    @override
    def upsert_index(
        self,
        data: Casebase[K, str],
    ) -> None:
        """Insert or replace entries in the embedding index.

        If no index exists yet, delegates to :meth:`put_index`.
        """
        if self._casebase is None:
            self.put_index(data)
            return

        if not data:
            return

        old_texts = self._obsolete_texts(upsert=data)

        if old_texts:
            self.conversion_func.delete_index(old_texts)

        self.conversion_func.upsert_index(data.values())
        self._casebase.update(data)

    @override
    def delete_index(
        self,
        data: Collection[K],
    ) -> None:
        """Remove entries by key from the embedding index."""
        if self._casebase is None or not data:
            return

        texts_to_delete = self._obsolete_texts(delete=data)

        if texts_to_delete:
            self.conversion_func.delete_index(texts_to_delete)

        for key in data:
            self._casebase.pop(key, None)

    @override
    def patch_index(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        """Apply inserts, replacements, and deletes to the embedding index."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_keys = normalized

        if self._casebase is None:
            if upsert:
                self.put_index(upsert)
            return

        texts_to_delete = self._obsolete_texts(upsert=upsert, delete=delete_keys)

        self.conversion_func.patch_index(
            upsert=upsert.values() if upsert else None,
            delete=texts_to_delete or None,
        )

        for key in delete_keys:
            self._casebase.pop(key, None)

        if upsert:
            self._casebase.update(upsert)

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], dict[K, S]]]:
        if not batches:
            return []

        resolved = resolve_casebases(batches, self._casebase)
        sim_maps = dispatch_batches(resolved, self.__call_queries__)

        return [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(resolved, sim_maps, strict=True)
        ]

    def __call_queries__(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
    ) -> Sequence[dict[K, S]]:
        case_texts = list(casebase.values())
        query_texts = list(queries)

        if self.query_conversion_func:
            case_vecs = self.conversion_func(case_texts)
            query_vecs = self.query_conversion_func(query_texts)
        else:
            all_texts = case_texts + query_texts
            all_vecs = self.conversion_func(all_texts)
            case_vecs = all_vecs[: len(case_texts)]
            query_vecs = all_vecs[len(case_texts) :]

        case_keys = list(casebase.keys())

        return [
            dict(
                zip(
                    case_keys,
                    self.sim_func([(cv, query_vec) for cv in case_vecs]),
                    strict=True,
                )
            )
            for query_vec in query_vecs
        ]


__all__ = ["embed"]
