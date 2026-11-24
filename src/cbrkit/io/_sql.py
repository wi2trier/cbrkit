"""The cases of a SQL query, which needs the `sql` extra."""

from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import sqlalchemy as sa

from .._util import keyed

__all__ = ["SQLQuery"]


@dataclass(slots=True, frozen=True, kw_only=True)
class SQLQuery(Mapping[Any, dict[str, Any]]):
    """The rows of any SQL query as a read-only case base.

    The query runs once when the case base is created, and its rows are kept in memory,
    keyed by the `key_column` or by position if it is None.
    The host creates and disposes the engine, and a connection of the host runs the query in its current transaction.
    The query may join tables, read views, aggregate, or compute columns,
    and needs no primary key or declared schema.
    `cbrkit.indexable.SQLAlchemy` is the writable storage of a single managed table instead.

    Args:
        engine: Engine or connection of the database, owned by the host,
            such as `sqlalchemy.create_engine("sqlite:///data.db")`.
        query: A raw SQL string or a SQLAlchemy Core construct such as `sqlalchemy.select`.
        params: Bind parameters of a raw SQL string, ignored for a Core construct.
        key_column: Column whose values key the rows, positions if None.

    Examples:
        >>> import sqlite3, tempfile
        >>> from pathlib import Path
        >>> db_path = Path(tempfile.mkdtemp()) / "test.db"
        >>> conn = sqlite3.connect(db_path)
        >>> _ = conn.execute("CREATE TABLE cases (id INTEGER, name TEXT)")
        >>> _ = conn.execute("INSERT INTO cases VALUES (1, 'a'), (2, 'b')")
        >>> conn.commit()
        >>> conn.close()
        >>> engine = sa.create_engine(f"sqlite:///{db_path}")
        >>> len(SQLQuery(engine=engine, query="SELECT * FROM cases"))
        2
        >>> cases = sa.Table("cases", sa.MetaData(), autoload_with=engine)
        >>> with engine.connect() as conn:
        ...     cb = SQLQuery(engine=conn, query=sa.select(cases).where(cases.c.id == 1))
        >>> engine.dispose()
        >>> cb[0]["name"]
        'a'
    """

    engine: sa.Engine | sa.Connection
    query: str | sa.Executable
    params: Mapping[str, Any] = field(default_factory=dict)
    key_column: str | None = None
    _data: Mapping[Any, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        stmt: sa.Executable = (
            sa.text(self.query) if isinstance(self.query, str) else self.query
        )
        connection = (
            nullcontext(self.engine)
            if isinstance(self.engine, sa.Connection)
            else self.engine.connect()
        )

        with connection as conn:
            rows = conn.execute(stmt, self.params or None).mappings().all()

        object.__setattr__(
            self, "_data", keyed((dict(row) for row in rows), self.key_column)
        )

    def __getitem__(self, key: Any) -> dict[str, Any]:
        return self._data[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)
