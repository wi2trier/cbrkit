from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from ..helpers import sim_map2ranking, singleton
from ..typing import (
    Casebase,
    Float,
    JsonEntry,
    SimMap,
)


class QueryResultStep[K, V, S: Float](BaseModel):
    """Result of a single query with similarities, ranking, and matched cases."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    similarities: SimMap[K, S]
    ranking: Sequence[K] = ()
    casebase: Casebase[K, V]
    query: V
    duration: float

    @model_validator(mode="before")
    @classmethod
    def _custom_validator(cls, data: dict[str, Any]) -> dict[str, Any]:
        assert len(data["similarities"]) == len(data["casebase"]), (
            "similarities and casebase must have equal length"
        )

        if not data.get("ranking"):
            data["ranking"] = tuple(sim_map2ranking(data["similarities"]))

        return data

    @property
    def casebase_similarities(self) -> Mapping[K, tuple[V, S]]:
        """Return a mapping of case keys to their case-similarity pairs."""
        return {
            key: (self.casebase[key], self.similarities[key]) for key in self.ranking
        }

    @property
    def similarity(self) -> S:
        """Return the similarity value when only a single case exists."""
        return singleton(self.similarities.values())

    @property
    def case(self) -> V:
        """Return the case value when only a single case exists."""
        return singleton(self.casebase.values())

    def remove_cases(self) -> "QueryResultStep[K, None, S]":
        """Return a copy of this result step with all case data removed."""
        return QueryResultStep[K, None, S](
            similarities=self.similarities,
            ranking=self.ranking,
            casebase={},
            query=None,
            duration=self.duration,
        )


class ResultStep[Q, C, V, S: Float](BaseModel):
    """Aggregated result step across multiple queries."""

    model_config = ConfigDict(frozen=True)
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonEntry
    duration: float

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        """Return the single query result when only one query exists."""
        return singleton(self.queries.values())

    @property
    def similarities(self) -> SimMap[C, S]:
        """Return the similarity map of the default query."""
        return self.default_query.similarities

    @property
    def ranking(self) -> Sequence[C]:
        """Return the case ranking of the default query."""
        return self.default_query.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        """Return the casebase of the default query."""
        return self.default_query.casebase

    @property
    def similarity(self) -> S:
        """Return the similarity value of the default query's single case."""
        return self.default_query.similarity

    @property
    def case(self) -> V:
        """Return the case value of the default query's single case."""
        return self.default_query.case

    def remove_cases(self) -> "ResultStep[Q, C, None, S]":
        """Return a copy of this result step with all case data removed."""
        return ResultStep[Q, C, None, S](
            queries={key: value.remove_cases() for key, value in self.queries.items()},
            metadata=self.metadata,
            duration=self.duration,
        )


class Result[Q, C, V, S: Float](BaseModel):
    """Complete result containing all pipeline steps."""

    model_config = ConfigDict(frozen=True)
    steps: list[ResultStep[Q, C, V, S]]
    duration: float

    @property
    def first_step(self) -> ResultStep[Q, C, V, S]:
        """Return the first step of the pipeline."""
        return self.steps[0]

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        """Return the last step of the pipeline."""
        return self.steps[-1]

    @property
    def metadata(self) -> JsonEntry:
        """Return the metadata from the final step."""
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        """Return the query results from the final step."""
        return self.final_step.queries

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        """Return the single query result from the final step."""
        return singleton(self.queries.values())

    @property
    def similarities(self) -> SimMap[C, S]:
        """Return the similarity map from the final step."""
        return self.final_step.similarities

    @property
    def ranking(self) -> Sequence[C]:
        """Return the case ranking from the final step."""
        return self.final_step.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        """Return the casebase from the final step."""
        return self.final_step.casebase

    @property
    def similarity(self) -> S:
        """Return the similarity value from the final step's single case."""
        return self.final_step.similarity

    @property
    def case(self) -> V:
        """Return the case value from the final step's single case."""
        return self.final_step.case

    def remove_cases(self) -> "Result[Q, C, None, S]":
        """Return a copy of this result with all case data removed."""
        return Result[Q, C, None, S](
            steps=[step.remove_cases() for step in self.steps],
            duration=self.duration,
        )


class CycleResult[Q, C, V, S: Float](BaseModel):
    """Combined results for all phases of a CBR cycle."""

    model_config = ConfigDict(frozen=True)
    retrieval: Result[Q, C, V, S]
    reuse: Result[Q, C, V, S]
    revise: Result[Q, C, V, S]
    retain: Result[Q, C, V, S]
    duration: float

    def remove_cases(self) -> "CycleResult[Q, C, None, S]":
        """Return a copy of this cycle result with all case data removed."""
        return CycleResult[Q, C, None, S](
            retrieval=self.retrieval.remove_cases(),
            reuse=self.reuse.remove_cases(),
            revise=self.revise.remove_cases(),
            retain=self.retain.remove_cases(),
            duration=self.duration,
        )
