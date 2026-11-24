import hashlib
import math
import re
import statistics
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from functools import partial
from typing import Any, Literal, TypedDict

import orjson
from pydantic import Field
from pydantic_core import to_jsonable_python

from ..model.graph import Graph as GraphValue
from ..schema import Model
from ..sim.dates import seconds

__all__ = [
    "Profile",
    "ValueType",
    "fingerprint",
    "profile",
]

type ValueType = Literal[
    "missing",
    "bool",
    "number",
    "date",
    "duration",
    "string",
    "list",
    "object",
    "graph",
    "mixed",
]

type Pattern = Literal["url", "email", "token"]

EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


class Profile(Model):
    """Statistics of the values of one element of a case base.

    Numbers are summarized by the 5th and 95th percentile, dates and durations
    likewise in days, and strings by their mean length and common pattern.
    """

    type: ValueType
    count: int = 0
    missing: int = 0
    distinct: int = 0
    samples: list[Any] = Field(default_factory=list)
    """The first distinct values in case base order."""
    low: float | None = None
    high: float | None = None
    mean_length: float | None = None
    pattern: Pattern | None = None
    """Shared by all strings, `token` means without whitespace."""
    attributes: dict[str, Profile] = Field(default_factory=dict)
    element: Profile | None = None
    node_type: str | None = None
    node: Profile | None = None
    node_types: dict[str, Profile] = Field(default_factory=dict)
    edge: Profile | None = None


def _is_date(value: str) -> bool:
    if len(value) < 10 or value[4] != "-":
        return False

    try:
        seconds(value)
    except ValueError:
        return False

    return True


def _kind(value: Any) -> ValueType:
    match value:
        case None:
            return "missing"
        case float() if math.isnan(value):
            return "missing"
        case bool():
            return "bool"
        case int() | float():
            return "number"
        case date():
            return "date"
        case timedelta():
            return "duration"
        case str():
            return "date" if _is_date(value) else "string"
        case GraphValue():
            return "graph"
        case Mapping():
            return "object"
        case list() | tuple() | set() | frozenset():
            return "list"
        case _:
            return "mixed"


def _key(value: Any) -> bytes:
    return orjson.dumps(
        value,
        default=partial(to_jsonable_python, fallback=str),
        option=orjson.OPT_SORT_KEYS,
    )


def _range(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]

    quantiles = statistics.quantiles(values, n=20, method="inclusive")

    return quantiles[0], quantiles[-1]


def _pattern(values: list[str]) -> Pattern | None:
    if all(value.startswith(("http://", "https://")) for value in values):
        return "url"

    if all(EMAIL.fullmatch(value) for value in values):
        return "email"

    if not any(char.isspace() for value in values for char in value):
        return "token"

    return None


def _common_kind(present: list[Any]) -> ValueType:
    """The kind of the values, mixed if they differ, and string if they are strings that look like dates."""
    kinds: set[ValueType] = {_kind(value) for value in present}

    if len(kinds) == 1:
        return kinds.pop()

    if kinds == {"string", "date"} and all(isinstance(value, str) for value in present):
        return "string"

    return "mixed"


class _Details(TypedDict, total=False):
    """The fields of a profile that depend on the kind of the values."""

    low: float
    high: float
    mean_length: float
    pattern: Pattern | None
    element: Profile
    attributes: dict[str, Profile]


def _details(
    kind: ValueType, present: list[Any], samples: int, type_attribute: str
) -> _Details:
    """The fields of a profile that depend on the kind of the values."""
    match kind:
        case "number":
            low, high = _range([float(value) for value in present])

            return {"low": low, "high": high}
        case "date" | "duration":
            low, high = _range([seconds(value) / 86400 for value in present])

            return {"low": low, "high": high}
        case "string":
            return {
                "mean_length": sum(len(value) for value in present) / len(present),
                "pattern": _pattern(present),
            }
        case "list":
            return {
                "element": profile(
                    (element for value in present for element in value),
                    samples,
                    type_attribute,
                )
            }
        case "object":
            names = dict.fromkeys(name for value in present for name in value)

            return {
                "attributes": {
                    str(name): profile(
                        (value.get(name) for value in present), samples, type_attribute
                    )
                    for name in names
                }
            }
        case _:
            return {}


def profile(
    values: Iterable[Any], samples: int = 20, type_attribute: str = "type"
) -> Profile:
    """Profiles values such as the cases of a case base.

    Args:
        values: Plain Python values, mappings and lists are profiled recursively and
            cbrkit graphs by their node and edge values.
        samples: Number of distinct sample values to keep.
        type_attribute: Attribute of node values that holds their type.

    Examples:
        >>> p = profile([{"price": 10, "tags": ["a"]}, {"price": 30}])
        >>> p.attributes["price"].type, p.attributes["price"].low, p.attributes["tags"].missing
        ('number', 11.0, 1)
    """
    items = list(values)
    present = [value for value in items if _kind(value) != "missing"]
    missing = len(items) - len(present)

    if not present:
        return Profile(type="missing", missing=missing)

    kind = _common_kind(present)

    if kind == "graph":
        return _graph_profile(present, missing, samples, type_attribute)

    distinct: dict[bytes, Any] = {}

    for value in present:
        distinct.setdefault(_key(value), value)

    return Profile(
        type=kind,
        count=len(present),
        missing=missing,
        distinct=len(distinct),
        samples=list(distinct.values())[:samples],
        **_details(kind, present, samples, type_attribute),
    )


def _graph_profile(
    graphs: list[GraphValue[Any, Any, Any, Any]],
    missing: int,
    samples: int,
    type_attribute: str,
) -> Profile:
    nodes = [node.value for graph in graphs for node in graph.nodes.values()]
    edges = [edge.value for graph in graphs for edge in graph.edges.values()]
    typed = all(
        isinstance(node, Mapping) and isinstance(node.get(type_attribute), str)
        for node in nodes
    )
    by_type: dict[str, list[Any]] = {}

    if typed and nodes:
        for node in nodes:
            by_type.setdefault(node[type_attribute], []).append(node)

    return Profile(
        type="graph",
        count=len(graphs),
        missing=missing,
        distinct=len(graphs),
        node_type=type_attribute if by_type else None,
        node=profile(nodes, samples, type_attribute),
        node_types={
            key: profile(values, samples, type_attribute)
            for key, values in by_type.items()
        },
        edge=profile(edges, samples, type_attribute),
    )


def _shape(p: Profile) -> dict[str, Any]:
    shape: dict[str, Any] = {"type": p.type}

    if p.attributes:
        shape["attributes"] = {
            name: _shape(value) for name, value in p.attributes.items()
        }

    if p.node_types:
        shape["node_types"] = {
            name: _shape(value) for name, value in p.node_types.items()
        }

    for name, child in (("element", p.element), ("node", p.node), ("edge", p.edge)):
        if child is not None:
            shape[name] = _shape(child)

    return shape


def fingerprint(p: Profile) -> str:
    """Hashes the shape of a profile, its element names and value types.

    Examples:
        >>> fingerprint(profile([{"a": 1}])) == fingerprint(profile([{"a": 2}]))
        True
    """
    return hashlib.sha256(_key(_shape(p))).hexdigest()
