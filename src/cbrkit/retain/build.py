from collections.abc import Sequence
from dataclasses import dataclass
from typing import override

from ..helpers import batchify_sim, get_logger, produce_factory, unpack_float
from ..typing import (
    AnySimFunc,
    Casebase,
    Float,
    MapAdaptationFunc,
    MaybeFactory,
    RetainerFunc,
    SimMap,
)

logger = get_logger(__name__)

__all__ = ["build", "dropout"]


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float](RetainerFunc[K, V, S]):
    """Builds a retainer that stores cases and assesses their quality.

    For each batch entry, the retainer applies the storage function
    to integrate the case into the casebase, then scores the updated
    casebase using the assessment function.

    Args:
        assess_func: Similarity function to evaluate case quality.
        storage_func: Decides whether and how to store a case.

    Examples:
        >>> import cbrkit
        >>> retainer = build(
        ...     assess_func=cbrkit.sim.generic.equality(),
        ...     storage_func=cbrkit.retain.static(
        ...         key_func=lambda keys: max(keys, default=-1) + 1,
        ...         casebase={},
        ...     ),
        ... )
    """

    assess_func: MaybeFactory[AnySimFunc[V, S]]
    storage_func: MaybeFactory[MapAdaptationFunc[K, V]]

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        storage_func = produce_factory(self.storage_func)

        # Step 1: Store (transformation)
        updated_batches: list[tuple[Casebase[K, V], V]] = []

        for casebase, query in batches:
            updated_casebase = storage_func(casebase, query)
            updated_batches.append((updated_casebase, query))

        # Step 2: Score (assessment)
        assess_func = batchify_sim(produce_factory(self.assess_func))

        flat_batches: list[tuple[V, V]] = []
        flat_index: list[tuple[int, K]] = []

        for idx, (casebase, query) in enumerate(updated_batches):
            for key, case in casebase.items():
                flat_index.append((idx, key))
                flat_batches.append((case, query))

        scores = assess_func(flat_batches)

        sim_maps: list[dict[K, S]] = [{} for _ in updated_batches]
        for (idx, key), score in zip(flat_index, scores, strict=True):
            sim_maps[idx][key] = score

        return [
            (casebase, sim_map)
            for (casebase, _), sim_map in zip(updated_batches, sim_maps, strict=True)
        ]


@dataclass(slots=True, frozen=True)
class dropout[K, V, S: Float](RetainerFunc[K, V, S]):
    """Filters retained cases based on their assessment scores.

    Wraps a retainer function and reverts newly added cases whose scores
    fall below a threshold. This separates the boolean retain/reject
    decision from the scoring and storage logic.

    Args:
        retainer_func: The retainer function to wrap.
        min_similarity: Minimum score for a newly added case to be kept.
        max_similarity: Maximum score for a newly added case to be kept.

    Examples:
        >>> import cbrkit
        >>> retainer = dropout(
        ...     retainer_func=cbrkit.retain.build(
        ...         assess_func=cbrkit.sim.generic.equality(),
        ...         storage_func=cbrkit.retain.static(
        ...             key_func=lambda keys: max(keys, default=-1) + 1,
        ...             casebase={},
        ...         ),
        ...     ),
        ...     min_similarity=0.5,
        ... )
    """

    retainer_func: RetainerFunc[K, V, S]
    min_similarity: float | None = None
    max_similarity: float | None = None

    @override
    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
        /,
    ) -> Sequence[tuple[Casebase[K, V], SimMap[K, S]]]:
        results = self.retainer_func(batches)
        filtered: list[tuple[Casebase[K, V], SimMap[K, S]]] = []

        for (orig_casebase, _), (new_casebase, sim_map) in zip(
            batches, results, strict=True
        ):
            new_keys = set(new_casebase) - set(orig_casebase)

            drop_keys: set[K] = set()

            for key in new_keys:
                score = unpack_float(sim_map[key])

                if self.min_similarity is not None and score < self.min_similarity:
                    drop_keys.add(key)
                elif self.max_similarity is not None and score > self.max_similarity:
                    drop_keys.add(key)

            if drop_keys:
                kept_casebase: Casebase[K, V] = {
                    k: v for k, v in new_casebase.items() if k not in drop_keys
                }
                kept_sim_map: SimMap[K, S] = {
                    k: v for k, v in sim_map.items() if k not in drop_keys
                }
                filtered.append((kept_casebase, kept_sim_map))
            else:
                filtered.append((new_casebase, sim_map))

        return filtered
