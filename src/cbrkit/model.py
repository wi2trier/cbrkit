from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .helpers import sim_map2ranking
from .typing import (
    Casebase,
    Float,
    JsonEntry,
    SimMap,
)


@dataclass(slots=True, frozen=True)
class QueryResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]
    query: V

    @property
    def similarity(self) -> S:
        if len(self.similarities) != 1:
            raise ValueError("The step contains multiple similarities.")

        return next(iter(self.similarities.values()))

    @property
    def case(self) -> V:
        if len(self.casebase) != 1:
            raise ValueError("The step contains multiple cases.")

        return next(iter(self.casebase.values()))

    @classmethod
    def build(
        cls, similarities: Mapping[K, S], full_casebase: Casebase[K, V], query: V
    ) -> "QueryResultStep[K, V, S]":
        ranking = sim_map2ranking(similarities)
        casebase = {key: full_casebase[key] for key in similarities}

        return cls(similarities, tuple(ranking), casebase, query)

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]
        del x["query"]

        return x


@dataclass(slots=True, frozen=True)
class ResultStep[Q, C, V, S: Float]:
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonEntry

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
        if len(self.queries) != 1:
            raise ValueError("The step contains multiple queries.")

        return next(iter(self.queries.values()))

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


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    steps: list[ResultStep[Q, C, V, S]]

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
        if len(self.queries) != 1:
            raise ValueError("The step contains multiple queries.")

        return next(iter(self.queries.values()))

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

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for step in x["steps"]:
            for item in step["queries"].values():
                del item["casebase"]
                del item["query"]

        return x
