"""Similarity measures for dates, datetimes and durations.

Values may be `datetime.date`, `datetime.datetime`, ISO 8601 strings, or
`datetime.timedelta` durations.
Naive datetimes are interpreted as UTC, and a date is the midnight of its day.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal, override

import numpy as np

from ..schema import serializable
from ..typing import ColumnSimFunc, NumpyArray

__all__ = ["DateLinear", "Temporal", "seconds"]

type Temporal = date | datetime | timedelta | str


def seconds(value: Temporal) -> float:
    """Converts a temporal value to seconds, since the epoch for points in time.

    Examples:
        >>> seconds("1970-01-02")
        86400.0
        >>> seconds(timedelta(minutes=1))
        60.0
    """
    match value:
        case timedelta():
            return value.total_seconds()
        case datetime():
            aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return aware.timestamp()
        case date():
            return datetime(value.year, value.month, value.day, tzinfo=UTC).timestamp()
        case str():
            return seconds(datetime.fromisoformat(value))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class DateLinear(ColumnSimFunc[Temporal]):
    """Linear similarity of two dates or durations based on their distance.

    Serialized durations are ISO 8601 strings such as `P30D` or numbers of seconds.

    Args:
        max_distance: Distance at which the similarity reaches 0.0.
        min_distance: Distance up to which the similarity is 1.0.

    Examples:
        >>> sim = DateLinear(max_distance=timedelta(days=10))
        >>> sim(date(2024, 1, 1), "2024-01-06")
        0.5
        >>> sim(timedelta(days=1), timedelta(days=21))
        0.0
    """

    max_distance: timedelta
    min_distance: timedelta = timedelta(0)
    kind: Literal["date_linear"] = field(default="date_linear", repr=False)

    def __post_init__(self) -> None:
        if self.max_distance < self.min_distance:
            raise ValueError("max_distance must not be smaller than min_distance")

    @override
    def __call__(self, case: Temporal, query: Temporal) -> float:
        dist = abs(seconds(case) - seconds(query))
        low, high = self.min_distance.total_seconds(), self.max_distance.total_seconds()

        if dist <= low:
            return 1.0

        if dist >= high:
            return 0.0

        return (high - dist) / (high - low)

    @override
    def columns(
        self, cases: Sequence[Temporal], queries: Sequence[Temporal]
    ) -> NumpyArray:
        x = np.fromiter(map(seconds, cases), np.float64, len(cases))
        y = np.fromiter(map(seconds, queries), np.float64, len(queries))
        dist = np.abs(x - y)
        low, high = self.min_distance.total_seconds(), self.max_distance.total_seconds()

        if high == low:
            return (dist <= low).astype(np.float64)

        return np.clip((high - dist) / (high - low), 0.0, 1.0)
