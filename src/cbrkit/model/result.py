"""Results of running the functions of the CBR cycle, see `cbrkit.cycle`."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

from pydantic import ConfigDict, computed_field

from .._util import fuse_rankings, sim_map2ranking
from ..schema import Model
from ..typing import Casebase, JsonEntry, SimMap
from .changes import Add, Changes, Update
from .explanation import Explanation
from .similarity import ChangeSim, SerializedFloat

__all__ = ["Hit", "QueryResult", "Result", "Step"]


class Hit[K, V, S: SerializedFloat](Model):
    """A case with its similarity to a query.

    `model_dump(mode="json")` gives plain JSON, with graphs and `cbrkit.helpers.frozendict` cases in their serialized form.

    Args:
        key: Key of the case in the casebase.
        value: The case.
        score: Similarity of the case to the query,
            with details such as the scores of the attributes or the issues of a critique.
        explanation: Breakdown of the similarity by attribute or graph element, if requested.
    """

    key: K
    value: V
    score: S
    explanation: Explanation | None = None


class QueryResult[K, V, S: SerializedFloat](Model):
    """Cases with their similarities to one query, ranked by similarity.

    Detailed similarities such as `AttributeValueSim` keep their details through JSON.

    Args:
        query: The query.
        casebase: The cases kept for the query.
        similarities: Similarity of every kept case to the query.
    """

    # the computed ranking is part of the dumped JSON, which validates back
    model_config = ConfigDict(extra="ignore")

    query: V
    casebase: Casebase[K, V]
    similarities: SimMap[K, S]

    @computed_field
    @property
    def ranking(self) -> tuple[K, ...]:
        """Keys of the cases by descending similarity."""
        return tuple(sim_map2ranking(self.similarities))

    def hits(self, limit: int | None = None) -> list[Hit[K, V, S]]:
        """The best cases with their similarities, by descending similarity.

        Args:
            limit: Maximal number of hits, all if None.

        Examples:
            >>> result = QueryResult(query="a", casebase={1: "a", 2: "b"}, similarities={1: 1.0, 2: 0.5})
            >>> result.hits(limit=1)
            [Hit(key=1, value='a', score=1.0, explanation=None)]
        """
        return [
            Hit.model_construct(
                key=key, value=self.casebase[key], score=self.similarities[key]
            )
            for key in sim_map2ranking(self.similarities, limit)
        ]

    @property
    def changes(self) -> Changes[K, V]:
        """The applied changes that stored the cases of a retain run, see `cbrkit.retain.Retainer`, empty otherwise."""
        return _changes([self])


@dataclass(slots=True, frozen=True)
class Step[Q, T](Mapping[Q, T]):
    """Outputs of one function for every query, a read-only mapping of every query key to its output.

    A function of the cycle yields a `QueryResult` per query.

    Args:
        outputs: Output of every query by its key.
        metadata: Name and configuration of the function.
        duration: Time spent in the function in seconds.

    Examples:
        >>> step = Step(outputs={"q": "response"}, metadata={}, duration=0.0)
        >>> step["q"], len(step)
        ('response', 1)
    """

    outputs: Mapping[Q, T]
    metadata: JsonEntry
    duration: float

    @override
    def __getitem__(self, query: Q) -> T:
        return self.outputs[query]

    @override
    def __iter__(self) -> Iterator[Q]:
        return iter(self.outputs)

    @override
    def __len__(self) -> int:
        return len(self.outputs)


@dataclass(slots=True, frozen=True)
class Result[Q, K, V, S: SerializedFloat](Mapping[Q, QueryResult[K, V, S]]):
    """Results of every function of the cycle, see `cbrkit.cycle.run`.

    It maps every query key to the result of the last function for that query.

    Args:
        steps: Results of the functions in the order in which they ran.
        duration: Time spent in all functions in seconds.

    Examples:
        >>> result = Result(
        ...     steps=(
        ...         Step(
        ...             outputs={"q": QueryResult(query="a", casebase={1: "a"}, similarities={1: 1.0})},
        ...             metadata={},
        ...             duration=0.0,
        ...         ),
        ...     ),
        ...     duration=0.0,
        ... )
        >>> list(result), len(result), result["q"].ranking
        (['q'], 1, (1,))
    """

    steps: Sequence[Step[Q, QueryResult[K, V, S]]]
    duration: float

    @override
    def __getitem__(self, query: Q) -> QueryResult[K, V, S]:
        return self.steps[-1][query]

    @override
    def __iter__(self) -> Iterator[Q]:
        return iter(self.steps[-1])

    @override
    def __len__(self) -> int:
        return len(self.steps[-1])

    def hits(self, limit: int | None = None) -> list[Hit[K, V, S | float]]:
        """The best cases for all queries as one ranking, such as of a search by several phrasings.

        The ranking of a single query keeps its similarities with their details,
        those of several queries are fused by reciprocal rank fusion, see `cbrkit.retrieve.fuse`,
        so a case that several queries find ranks higher.

        Args:
            limit: Maximal number of hits, all if None.

        Examples:
            >>> step = Step(
            ...     outputs={
            ...         "a": QueryResult(query="a", casebase={1: "a", 2: "b"}, similarities={1: 0.9, 2: 0.1}),
            ...         "b": QueryResult(query="b", casebase={1: "a", 2: "b"}, similarities={1: 0.2, 2: 0.8}),
            ...     },
            ...     metadata={},
            ...     duration=0.0,
            ... )
            >>> [hit.key for hit in Result(steps=(step,), duration=0.0).hits(limit=1)]
            [1]
        """
        if len(self) == 1:
            return [
                Hit.model_construct(key=hit.key, value=hit.value, score=hit.score)
                for result in self.values()
                for hit in result.hits(limit)
            ]

        casebase, scores = fuse_rankings(
            ((result.casebase, result.similarities) for result in self.values()),
            limit=limit,
        )

        return [
            Hit.model_construct(key=key, value=casebase[key], score=score)
            for key, score in scores.items()
        ]

    @property
    def changes(self) -> Changes[K, V]:
        """The applied changes that stored the cases of a retain run for every query, see `QueryResult.changes`."""
        return _changes(self.values())


def _changes[K, V](results: Iterable[QueryResult[K, V, Any]]) -> Changes[K, V]:
    items: dict[K, Add[K, V] | Update[K, V]] = {}

    for result in results:
        for key, sim in result.similarities.items():
            if not isinstance(sim, ChangeSim):
                continue

            items[key] = sim.change(key, result.casebase[key])

    return Changes(items=list(items.values()), applied=True)
