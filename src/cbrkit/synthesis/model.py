from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from ..helpers import singleton
from ..typing import JsonEntry


class QueryResultStep[T](BaseModel):
    """Synthesis result for a single query."""

    model_config = ConfigDict(frozen=True)
    response: T
    duration: float


class ResultStep[Q, T](BaseModel):
    """Aggregated synthesis result step across multiple queries."""

    model_config = ConfigDict(frozen=True)
    queries: Mapping[Q, QueryResultStep[T]]
    metadata: JsonEntry
    duration: float

    @property
    def default_query(self) -> QueryResultStep[T]:
        """Return the single query result, raising if there are multiple."""
        return singleton(self.queries.values())

    @property
    def response(self) -> T:
        """Return the response of the default query."""
        return self.default_query.response


class Result[Q, T](BaseModel):
    """Complete synthesis result containing all steps."""

    model_config = ConfigDict(frozen=True)
    steps: list[ResultStep[Q, T]]
    duration: float

    @property
    def final_step(self) -> ResultStep[Q, T]:
        """Return the last step of the synthesis pipeline."""
        return self.steps[-1]

    @property
    def metadata(self) -> JsonEntry:
        """Return metadata from the final synthesis step."""
        return self.final_step.metadata

    @property
    def queries(self) -> Mapping[Q, QueryResultStep[T]]:
        """Return the query results from the final synthesis step."""
        return self.final_step.queries

    @property
    def default_query(self) -> QueryResultStep[T]:
        """Return the single query result from the final step."""
        return singleton(self.queries.values())

    @property
    def response(self) -> T:
        """Return the response from the final synthesis step."""
        return self.final_step.response
