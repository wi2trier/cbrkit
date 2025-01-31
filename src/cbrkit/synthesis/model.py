from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from ..helpers import singleton
from ..typing import JsonEntry


class QueryResultStep[T](BaseModel):
    model_config = ConfigDict(frozen=True)
    response: T


class ResultStep[Q, T](BaseModel):
    model_config = ConfigDict(frozen=True)
    queries: Mapping[Q, QueryResultStep[T]]
    metadata: JsonEntry

    @property
    def default_query(self) -> QueryResultStep[T]:
        return singleton(self.queries.values())

    @property
    def response(self) -> T:
        return self.default_query.response


class Result[Q, T](BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: list[ResultStep[Q, T]]

    @property
    def final_step(self) -> ResultStep[Q, T]:
        return self.steps[-1]

    @property
    def metadata(self) -> JsonEntry:
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[T]]:
        return self.final_step.queries

    @property
    def default_query(self) -> QueryResultStep[T]:
        return singleton(self.queries.values())

    @property
    def response(self) -> T:
        return self.final_step.response
