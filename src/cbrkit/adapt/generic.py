from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, override

from ..helpers import batchify
from ..schema import serializable
from ..sim import Measure
from ..typing import AdaptationFunc, BatchAdaptationFunc, Float
from ._nested import NestedAdaptation

__all__ = [
    "Null",
    "Pipe",
    "Union",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Pipe[V](BatchAdaptationFunc[V]):
    """Chain multiple adaptation functions together.

    Every step adapts all pairs in one batch, so a batch adaptation such as `cbrkit.adapt.llm.Rewrite` sends them at once.

    Args:
        adaptations: Adaptation functions to apply in order.
        measure: Similarity of a case to the query, which keeps only steps that improve it if given.
        similarity_delta: Minimal improvement of the similarity that keeps a step.
        stop: Stop adapting a case after its first step that is not kept instead of continuing with the next one.

    Examples:
        >>> func = Pipe(adaptations=[
        ...     lambda x, y: x + y,
        ...     lambda x, y: x * y,
        ... ])
        >>> func([(2, 3)])
        [15]
        >>> func = Pipe(
        ...     adaptations=[lambda x, y: x + 10, lambda x, y: x + 1],
        ...     measure=lambda x, y: 1 / (1 + abs(x - y)),
        ... )
        >>> func([(2, 3), (0, 20)])
        [3, 11]
    """

    adaptations: Sequence[NestedAdaptation[V]]
    measure: Measure[V, Float] | None = None
    similarity_delta: float = 0.0
    stop: bool = False
    kind: Literal["pipe"] = field(default="pipe", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]], /) -> Sequence[V]:
        cases = [case for case, _ in batches]
        queries = [query for _, query in batches]
        measure = None if self.measure is None else batchify(self.measure)
        scores = [] if measure is None else [float(sim) for sim in measure(batches)]
        active = list(range(len(batches)))

        for func in self.adaptations:
            if not active:
                break

            adapted = batchify(func)([(cases[idx], queries[idx]) for idx in active])

            if measure is None:
                for idx, case in zip(active, adapted, strict=True):
                    cases[idx] = case

                continue

            sims = measure(
                [
                    (case, queries[idx])
                    for idx, case in zip(active, adapted, strict=True)
                ]
            )
            kept: list[int] = []

            for idx, case, sim in zip(active, adapted, sims, strict=True):
                if float(sim) >= scores[idx] + self.similarity_delta:
                    cases[idx], scores[idx] = case, float(sim)
                    kept.append(idx)
                elif not self.stop:
                    kept.append(idx)

            active = kept

        return cases


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Null[V](AdaptationFunc[V]):
    """Perform a null adaptation and return the original case or query value.

    Args:
        target: Whether to return the case or the query value.
        copy: Whether to copy the value before returning it.

    Examples:
        >>> func = Null()
        >>> func(2, 3)
        2
        >>> Null(target="query")(2, 3)
        3
    """

    target: Literal["case", "query"] = "case"
    copy: bool = False
    kind: Literal["null"] = field(default="null", repr=False)

    @override
    def __call__(self, case: V, query: V) -> V:
        value = case if self.target == "case" else query

        return deepcopy(value) if self.copy else value


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Union(AdaptationFunc[Sequence[Any]]):
    """The elements of the case followed by those of the query that the case lacks.

    Examples:
        >>> Union()([1, 2], [2, 3])
        [1, 2, 3]
    """

    kind: Literal["union"] = field(default="union", repr=False)

    @override
    def __call__(self, case: Sequence[Any], query: Sequence[Any]) -> list[Any]:
        return [*case, *(value for value in query if value not in case)]
