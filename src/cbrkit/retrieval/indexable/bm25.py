"""BM25 retriever wrapper for :class:`cbrkit.sim.embed.bm25`."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import cast, override

import bm25s

from ...helpers import dispatch_batches
from ...indexable._common import _normalize_patch_keys
from ...sim.embed import bm25 as bm25_embed
from ...typing import Casebase, IndexableFunc, RetrieverFunc, SimMap
from ._common import _normalize_results, resolve_casebases


@dataclass(slots=True)
class bm25[K](
    RetrieverFunc[K, str, float],
    IndexableFunc[Casebase[K, str], Collection[K]],
):
    """BM25 retriever based on bm25s.

    Delegates BM25 model management to a
    :class:`~cbrkit.sim.embed.bm25` instance and performs
    BM25 scoring for retrieval.

    Args:
        conversion_func: BM25 sparse embedding function
            (from :mod:`cbrkit.sim.embed`).
        normalize_scores: If `True` (default), apply per-query min-max
            normalization to BM25 scores. If `False`, return raw BM25
            scores.

    Pass an empty casebase to `__call__` to use the pre-indexed casebase.
    """

    conversion_func: bm25_embed
    normalize_scores: bool = True
    _keys: list[K] | None = field(default=None, init=False, repr=False)

    @override
    def has_index(self) -> bool:
        """Return whether a BM25 index has been created."""
        return self._keys is not None

    @property
    @override
    def index(self) -> Casebase[K, str]:
        """Return the indexed casebase."""
        corpus = self.conversion_func._corpus
        if self._keys is None or corpus is None:
            return {}
        return dict(zip(self._keys, corpus, strict=True))

    @override
    def put_index(
        self,
        data: Casebase[K, str],
    ) -> None:
        """Rebuild BM25 index."""
        self._keys = list(data.keys())
        self.conversion_func.put_index(data.values())

    @override
    def upsert_index(
        self,
        data: Casebase[K, str],
    ) -> None:
        """Merge new data with existing casebase and rebuild index."""
        if self._keys is None:
            self.put_index(data)
            return

        merged = dict(self.index)
        merged.update(data)
        self.put_index(merged)

    @override
    def delete_index(
        self,
        data: Collection[K],
    ) -> None:
        """Remove keys from the casebase and rebuild index."""
        if self._keys is None:
            return

        remove = set(data)
        remaining = {k: v for k, v in self.index.items() if k not in remove}
        self.put_index(remaining)

    @override
    def patch_index(
        self,
        upsert: Casebase[K, str] | None = None,
        delete: Collection[K] | None = None,
    ) -> None:
        """Apply inserts, replacements, and deletes to the BM25 index."""
        normalized = _normalize_patch_keys(upsert, delete)

        if normalized is None:
            return

        _, delete_keys = normalized
        merged = dict(self.index)

        for key in delete_keys:
            merged.pop(key, None)

        if upsert:
            merged.update(upsert)

        self.put_index(merged)

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, str], str]],
    ) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
        indexed = self.index or None
        resolved = resolve_casebases(batches, indexed)

        def call_queries(
            queries: Sequence[str],
            casebase: Casebase[K, str],
        ) -> Sequence[dict[K, float]]:
            """Dispatch queries to the BM25 retriever with the indexed casebase."""
            return self.__call_queries__(queries, casebase, indexed)

        sim_maps = dispatch_batches(resolved, call_queries)

        results = [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(resolved, sim_maps, strict=True)
        ]

        return _normalize_results(results, self.normalize_scores)

    def __call_queries__(
        self,
        queries: Sequence[str],
        casebase: Casebase[K, str],
        indexed: Casebase[K, str] | None,
    ) -> Sequence[dict[K, float]]:
        if (
            self.conversion_func._retriever is not None
            and indexed is not None
            and casebase is indexed
        ):
            retriever = self.conversion_func._retriever
        else:
            retriever = self.conversion_func._build_retriever(casebase.values())

        queries_tokens = bm25s.tokenize(
            cast(list[str], queries),
            stemmer=self.conversion_func._stemmer,
            stopwords=self.conversion_func._stopwords,
        )

        results, scores = retriever.retrieve(
            queries_tokens,
            sorted=False,
            k=len(casebase),
        )

        key_index = dict(enumerate(casebase))

        return [
            {
                key_index[case_id]: float(score)
                for case_id, score in zip(
                    results[query_id], scores[query_id], strict=True
                )
            }
            for query_id in range(len(queries))
        ]


__all__ = ["bm25"]
