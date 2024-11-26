from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..helpers import (
    similarities2ranking,
)
from ..typing import (
    Casebase,
    Float,
    JsonDict,
    SimMap,
)


@dataclass(slots=True, frozen=True)
class QueryResultStep[K, V, S: Float]:
    similarities: SimMap[K, S]
    ranking: Sequence[K]
    casebase: Casebase[K, V]

    @classmethod
    def build(
        cls, similarities: Mapping[K, S], full_casebase: Casebase[K, V]
    ) -> "QueryResultStep[K, V, S]":
        ranking = similarities2ranking(similarities)
        casebase = {key: full_casebase[key] for key in similarities}

        return cls(similarities, tuple(ranking), casebase)

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)
        del x["casebase"]

        return x


@dataclass(slots=True, frozen=True)
class ResultStep[Q, C, V, S: Float]:
    queries: Mapping[Q, QueryResultStep[C, V, S]]
    metadata: JsonDict

    @property
    def default_query(self) -> QueryResultStep[C, V, S]:
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


@dataclass(slots=True, frozen=True)
class Result[Q, C, V, S: Float]:
    steps: list[ResultStep[Q, C, V, S]]

    @property
    def final_step(self) -> ResultStep[Q, C, V, S]:
        return self.steps[-1]

    @property
    def metadata(self) -> JsonDict:
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[C, V, S]]:
        return self.final_step.queries

    @property
    def similarities(self) -> SimMap[C, S]:
        return self.final_step.similarities

    @property
    def ranking(self) -> Sequence[C]:
        return self.final_step.ranking

    @property
    def casebase(self) -> Casebase[C, V]:
        return self.final_step.casebase

    def as_dict(self) -> dict[str, Any]:
        x = asdict(self)

        for step in x["steps"]:
            for item in step["queries"].values():
                del item["casebase"]

        return x
