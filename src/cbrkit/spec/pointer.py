"""JSON pointers into trees of mappings, dataclasses, and pydantic models, such as measures, profiles, and their JSON."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

from pydantic import BaseModel

from ..helpers import replace_attributes

__all__ = ["escape", "get", "parent", "put", "segments"]


def escape(segment: str) -> str:
    """Escapes a segment of a JSON pointer.

    Examples:
        >>> escape("a/b~c")
        'a~1b~0c'
    """
    return segment.replace("~", "~0").replace("/", "~1")


def segments(pointer: str) -> list[str]:
    """The unescaped segments of a JSON pointer.

    Examples:
        >>> segments("/attributes/a~1b~0c")
        ['attributes', 'a/b~c']
    """
    return [
        segment.replace("~1", "/").replace("~0", "~")
        for segment in pointer.split("/")[1:]
    ]


def parent(pointer: str) -> tuple[str, str] | None:
    """The pointer of the parent and the unescaped last segment, None for the root.

    Examples:
        >>> parent("/attributes/a~1b"), parent("")
        (('/attributes', 'a/b'), None)
    """
    if not pointer:
        return None

    head, _, last = pointer.rpartition("/")

    return head, segments(f"/{last}")[0]


def _child(node: object, segment: str) -> Any:
    """The value of a mapping or of a field of a dataclass or model by a segment, None if there is none."""
    if isinstance(node, Mapping):
        return node.get(segment)

    if isinstance(node, BaseModel):
        return getattr(node, segment) if segment in type(node).model_fields else None

    if is_dataclass(node) and segment in {f.name for f in fields(node)}:
        return getattr(node, segment)

    return None


def get(tree: object, pointer: str) -> Any:
    """The value at a pointer, None if it does not resolve.

    Examples:
        >>> get({"a": [1], "b": {"c": 2}}, "/b/c"), get({"a": 1}, "/b/c")
        (2, None)
    """
    node: Any = tree

    for segment in segments(pointer):
        node = _child(node, segment)

    return node


def put[T](tree: T, pointer: str, value: Any) -> T:
    """A copy of the tree with the value at a pointer, which the root pointer replaces.

    Examples:
        >>> put({"a": {"b": 1}}, "/a/b", 2)
        {'a': {'b': 2}}
    """

    def update(node: Any, parts: list[str]) -> Any:
        if not parts:
            return value

        head, *rest = parts

        return replace_attributes(node, {head: update(_child(node, head), rest)})

    return update(tree, segments(pointer))
