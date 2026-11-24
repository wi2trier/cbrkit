import operator
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, override

import numpy as np

from .._util import fold
from ..schema import Unit, serializable
from ..typing import ColumnSimFunc, NumpyArray

__all__ = [
    "Equality",
    "Static",
    "Table",
    "TypeEquality",
]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Table(ColumnSimFunc[Hashable]):
    """Lookup table of value pairs, equal values have a similarity of 1.0 unless listed.

    Args:
        entries: Pairs of values with their similarity.
        default: Similarity of pairs that are not listed, such pairs raise a `KeyError` if None.
        symmetric: Whether an entry also applies to the reversed pair.
        case_sensitive: Whether strings are compared case-sensitively.

    Examples:
        >>> sim = Table(entries=[("a", "b", 0.5), ("b", "c", 0.7)], default=0.0)
        >>> sim("b", "a"), sim("a", "c"), sim("c", "c")
        (0.5, 0.0, 1.0)
        >>> Table(entries=[("Red", "Orange", 0.5)], case_sensitive=False)("red", "orange")
        0.5
    """

    entries: Sequence[tuple[Hashable, Hashable, Unit]]
    default: Unit | None = None
    symmetric: bool = True
    case_sensitive: bool = True
    kind: Literal["table"] = field(default="table", repr=False)
    _lookup: dict[tuple[Hashable, Hashable], float] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        lookup: dict[tuple[Hashable, Hashable], float] = {}

        for case, query, value in self.entries:
            pair = (fold(case, self.case_sensitive), fold(query, self.case_sensitive))
            lookup[pair] = value

            if self.symmetric:
                lookup[pair[1], pair[0]] = value

        object.__setattr__(self, "_lookup", lookup)

    @override
    def __call__(self, case: Hashable, query: Hashable) -> float:
        pair = (fold(case, self.case_sensitive), fold(query, self.case_sensitive))

        if pair in self._lookup:
            return self._lookup[pair]

        if pair[0] == pair[1]:
            return 1.0

        if self.default is None:
            raise KeyError(f"Pair {pair} not in the table")

        return self.default

    @override
    def columns(
        self, cases: Sequence[Hashable], queries: Sequence[Hashable]
    ) -> NumpyArray:
        return np.fromiter(map(self, cases, queries), np.float64, len(cases))


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Equality(ColumnSimFunc[Any]):
    """Equality similarity function. Returns 1.0 if the two values are equal, 0.0 otherwise.

    Args:
        case_sensitive: If False, strings are compared case-insensitively.
        normalize_whitespace: If True, leading, trailing and repeated whitespace of strings is ignored.

    Examples:
        >>> sim = Equality()
        >>> sim("b", "a")
        0.0
        >>> sim("a", "a")
        1.0
        >>> Equality(case_sensitive=False, normalize_whitespace=True)(" New  York", "new york")
        1.0
    """

    case_sensitive: bool = True
    normalize_whitespace: bool = False
    kind: Literal["equality"] = field(default="equality", repr=False)

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, str) and self.normalize_whitespace:
            value = " ".join(value.split())

        return fold(value, self.case_sensitive)

    @override
    def __call__(self, case: Any, query: Any) -> float:
        return 1.0 if self._normalize(case) == self._normalize(query) else 0.0

    @override
    def columns(self, cases: Sequence[Any], queries: Sequence[Any]) -> NumpyArray:
        if not self.case_sensitive or self.normalize_whitespace:
            cases = [self._normalize(value) for value in cases]
            queries = [self._normalize(value) for value in queries]

        equal = map(operator.eq, cases, queries)

        return np.fromiter(equal, np.bool_, len(cases)).astype(np.float64)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class TypeEquality(ColumnSimFunc[Any]):
    """Type equality similarity function. Returns 1.0 if the two values have the same type, 0.0 otherwise.

    Examples:
        >>> sim = TypeEquality()
        >>> sim("b", "a")
        1.0
        >>> sim("a", 1)
        0.0
    """

    kind: Literal["type_equality"] = field(default="type_equality", repr=False)

    @override
    def __call__(self, case: Any, query: Any) -> float:
        return 1.0 if type(case) is type(query) else 0.0

    @override
    def columns(self, cases: Sequence[Any], queries: Sequence[Any]) -> NumpyArray:
        same = (
            type(case) is type(query)
            for case, query in zip(cases, queries, strict=True)
        )

        return np.fromiter(same, np.bool_, len(cases)).astype(np.float64)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Static(ColumnSimFunc[Any]):
    """Static similarity function. Returns a constant value for all pairs.

    Args:
        value: The constant similarity.

    Examples:
        >>> sim = Static(value=0.5)
        >>> sim("b", "a")
        0.5
        >>> sim("a", "a")
        0.5
    """

    value: Unit
    kind: Literal["static"] = field(default="static", repr=False)

    @override
    def __call__(self, case: Any, query: Any) -> float:
        return self.value

    @override
    def columns(self, cases: Sequence[Any], queries: Sequence[Any]) -> NumpyArray:
        return np.full(len(cases), self.value, np.float64)
