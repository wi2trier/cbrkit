"""Shared helpers and mixins for indexable retrievers."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...helpers import batchify_sim
from ...sim.embed import default_score_func
from ...typing import (
    BatchConversionFunc,
    Casebase,
    NumpyArray,
    SimMap,
)


def resolve_casebases[K, V](
    batches: Sequence[tuple[Casebase[K, V], V]],
    indexed_casebase: Casebase[K, V] | None,
) -> list[tuple[Casebase[K, V], V]]:
    """Resolve casebases for indexable retrievers.

    Empty casebases are treated as an explicit signal to use indexed retrieval mode.
    In indexed mode, empty casebases are replaced with the previously indexed casebase.
    """
    if indexed_casebase is None:
        if any(len(casebase) == 0 for casebase, _ in batches):
            raise ValueError(
                "Indexed retrieval was requested with an empty casebase, but no index is available. "
                "Call put_index() first."
            )

        return list(batches)

    return [
        (indexed_casebase if len(casebase) == 0 else casebase, query)
        for casebase, query in batches
    ]


def _normalize_results[K](
    results: Sequence[tuple[Casebase[K, str], SimMap[K, float]]],
    enabled: bool,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Apply per-query min-max normalization if enabled.

    When all scores in a query coincide (e.g. single-hit queries), every
    entry is mapped to 1.0 — the best score — rather than 0.0, which
    would silently mark a perfect single match as a non-match.
    """
    if not enabled:
        return results

    normalized: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for cb, sm in results:
        if not sm:
            normalized.append((cb, sm))
            continue

        mn = min(sm.values())
        mx = max(sm.values())

        if mn == mx:
            normalized.append((cb, {k: 1.0 for k in sm}))
            continue

        spread = mx - mn
        normalized.append((cb, {k: (v - mn) / spread for k, v in sm.items()}))

    return normalized


def _brute_force_dense_search[K](
    queries: Sequence[str],
    casebase: Casebase[K, str],
    conversion_func: BatchConversionFunc[str, NumpyArray],
    query_conversion_func: BatchConversionFunc[str, NumpyArray] | None,
) -> Sequence[tuple[Casebase[K, str], SimMap[K, float]]]:
    """Shared brute-force dense vector search for non-indexed casebases."""
    keys = list(casebase.keys())
    values = list(casebase.values())

    case_vecs = conversion_func(values)
    embed_func = query_conversion_func or conversion_func
    query_vecs = embed_func(list(queries))

    sim_func = batchify_sim(default_score_func)

    results: list[tuple[Casebase[K, str], SimMap[K, float]]] = []

    for qvec in query_vecs:
        sims = sim_func([(cv, qvec) for cv in case_vecs])
        results.append(
            (
                dict(casebase),
                dict(zip(keys, sims, strict=True)),
            )
        )

    return results


@dataclass(slots=True, kw_only=True)
class RrfMixin:
    """Reciprocal Rank Fusion parameters shared across hybrid retrievers.

    Keyword-only so subclass dataclasses can keep positional fields
    without conflicting with the defaults declared here.
    """

    rrf_k: int = 60
    """Smoothing parameter in the RRF denominator: ``1 / (rrf_k + rank)``."""

    rrf_weights: tuple[float, float] = (0.7, 0.3)
    """Weights for ``(dense, sparse)`` rankings in the fusion sum."""


def reciprocal_rank_fusion[K, V](
    rankings: Sequence[Iterable[tuple[K, V]]],
    weights: Sequence[float],
    rrf_k: int,
) -> tuple[dict[K, float], dict[K, V]]:
    """Combine multiple ranked lists into RRF-fused scores.

    Each ranking is an iterable of ``(key, value)`` pairs in ranked
    order (rank 1 first).  Returns ``(scores, values)`` where
    ``scores[key] = Σ wᵢ / (rrf_k + rankᵢ)`` and ``values[key]`` is the
    first value seen for that key across rankings.
    """
    assert len(rankings) == len(weights)
    scores: dict[K, float] = {}
    values: dict[K, V] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, (key, value) in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            values.setdefault(key, value)
    return scores, values


__all__ = [
    "resolve_casebases",
    "_normalize_results",
    "_brute_force_dense_search",
    "RrfMixin",
    "reciprocal_rank_fusion",
]
