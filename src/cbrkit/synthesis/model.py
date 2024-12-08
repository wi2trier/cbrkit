from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from ..helpers import singleton
from ..typing import JsonEntry


@dataclass(slots=True, frozen=True)
class QueryResultStep[T]:
    response: T

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ResultStep[Q, T]:
    queries: Mapping[Q, QueryResultStep[T]]
    metadata: JsonEntry

    @property
    def default_query(self) -> QueryResultStep[T]:
        return singleton(self.queries.values())

    @property
    def response(self) -> T:
        return self.default_query.response


@dataclass(slots=True, frozen=True)
class Result[Q, T]:
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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
