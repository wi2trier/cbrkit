import csv
import fnmatch
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from ...typing import FilePath, SimFunc
from ..generic import static_table


def table(
    entries: Sequence[tuple[str, str, float]]
    | Mapping[tuple[str, str], float]
    | FilePath,
    symmetric: bool = True,
    default: float = 0.0,
) -> SimFunc[str, float]:
    """Allows to import a similarity values from a table.

    Args:
        entries: Sequence[tuple[a, b, sim(a, b)]
        symmetric: If True, the table is assumed to be symmetric, i.e. sim(a, b) = sim(b, a)
        default: Default similarity value for batches not in the table

    Examples:
        >>> sim = table(
        ...     [("a", "b", 0.5), ("b", "c", 0.7)],
        ...     symmetric=True,
        ...     default=0.0
        ... )
        >>> sim("b", "a")
        0.5
        >>> sim("a", "c")
        0.0
    """
    if isinstance(entries, str | Path):
        if isinstance(entries, str):
            entries = Path(entries)

        if entries.suffix != ".csv":
            raise NotImplementedError()

        with entries.open() as f:
            reader = csv.reader(f)
            parsed_entries = [(x, y, float(z)) for x, y, z in reader]

    else:
        parsed_entries = entries

    return static_table(parsed_entries, symmetric=symmetric, default=default)


@dataclass(slots=True, frozen=True)
class regex(SimFunc[str, float]):
    """Compares a case x to a query y, written as a regular expression. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Examples:
        >>> sim = regex()
        >>> sim("Test1", "T.st[0-9]")
        1.0
        >>> sim("Test2", "T.st[3-6]")
        0.0
    """

    @override
    def __call__(self, x: str, y: str) -> float:
        regex = re.compile(y)
        return 1.0 if regex.match(x) else 0.0


@dataclass(slots=True, frozen=True)
class glob(SimFunc[str, float]):
    """Compares a case x to a query y, written as a glob pattern, which can contain wildcards. If the case matches the query, the similarity is 1.0, otherwise 0.0.

    Args:
        case_sensitive: If True, the comparison is case-sensitive
    Examples:
        >>> sim = glob()
        >>> sim("Test1", "Test?")
        1.0
        >>> sim("Test2", "Test[3-9]")
        0.0
    """

    case_sensitive: bool = False

    @override
    def __call__(self, x: str, y: str) -> float:
        comparison_func = (
            fnmatch.fnmatchcase if self.case_sensitive else fnmatch.fnmatch
        )
        return 1.0 if comparison_func(x, y) else 0.0
