"""Backend-agnostic filter AST for filterable storage backends.

Filter expressions are compiled per backend (`postgresql` → SQLAlchemy
`ColumnElement[bool]`, `lancedb` → SQL string) so the same predicate
travels unchanged from the host to the storage layer.
"""

from collections.abc import Collection
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Eq:
    """Equality predicate: ``column = value``."""

    column: str
    value: int | str | bool


@dataclass(slots=True, frozen=True)
class In:
    """Set-membership predicate: ``column IN (...)``."""

    column: str
    values: Collection[int | str]


@dataclass(slots=True, frozen=True)
class Like:
    """Pattern predicate: ``column LIKE pattern``."""

    column: str
    pattern: str
    escape: str | None = None


@dataclass(slots=True, frozen=True)
class And:
    """Conjunction of filters."""

    filters: tuple["Filter", ...]


@dataclass(slots=True, frozen=True)
class Or:
    """Disjunction of filters."""

    filters: tuple["Filter", ...]


@dataclass(slots=True, frozen=True)
class Not:
    """Negation of a filter."""

    inner: "Filter"


@dataclass(slots=True, frozen=True)
class Raw:
    """Backend-native escape hatch.

    The string is passed verbatim to the backend (LanceDB SQL string,
    SQLAlchemy ``sa.text``).  Never build this from untrusted input.
    """

    sql: str


type Filter = Eq | In | Like | And | Or | Not | Raw


__all__ = [
    "Eq",
    "In",
    "Like",
    "And",
    "Or",
    "Not",
    "Raw",
    "Filter",
]
