from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from ..helpers import sim_map2ranking, singleton
from ..typing import (
    Casebase,
    Float,
    JsonEntry,
    SimMap,
)


class QueryResultStep[K, V, S: Float](BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]
    query: V
    duration: float

    @property
    def casebase_similarities(self) -> Mapping[K, tuple[V, S]]:
        return {
            key: (self.casebase[key], self.similarities[key]) for key in self.ranking
        }

    @property
    def similarity(self) -> S:
        return singleton(self.similarities.values())

    @property
    def case(self) -> V:
        return singleton(self.casebase.values())

    @classmethod
    def build(
        cls,
        similarities: Mapping[K, S],
        unfiltered_casebase: Casebase[K, V],
        query: V,
        duration: float,
    ) -> "QueryResultStep[K, V, S]":
        ranking = sim_map2ranking(similarities)
        filtered_casebase = {key: unfiltered_casebase[key] for key in similarities}

        return cls(
            similarities=similarities,
            ranking=tuple(ranking),
            casebase=filtered_casebase,
            query=query,
            duration=duration,
        )


class ResultStep[Q, C, V, S: Float](BaseModel):
    model_config = ConfigDict(frozen=True)
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonEntry
    duration: float

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        return singleton(self.queries.values())

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.default_query.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.default_query.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.default_query.casebase

    @property
    def similarity(self) -> S:
        return self.default_query.similarity

    @property
    def case(self) -> V:
        return self.default_query.case


class Result[Q, C, V, S: Float](BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: list[ResultStep[Q, C, V, S]]
    duration: float

    @property
    def first_step(self) -> ResultStep[Q, C, V, S]:
        return self.steps[0]

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        return self.steps[-1]

    @property
    def metadata(self) -> JsonEntry:
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        return self.final_step.queries

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        return singleton(self.queries.values())

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.final_step.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.final_step.casebase

    @property
    def similarity(self) -> S:
        return self.final_step.similarity

    @property
    def case(self) -> V:
        return self.final_step.case


class CycleResult[Q, C, V, S: Float](BaseModel):
    model_config = ConfigDict(frozen=True)
    retrieval: Result[Q, C, V, S]
    reuse: Result[Q, C, V, S]
    duration: float

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        if len(self.reuse.steps) > 0:
            return self.reuse.final_step

        return self.retrieval.final_step
