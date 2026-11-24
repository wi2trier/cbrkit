"""Similarity values with details, the subclasses of `cbrkit.typing.Sim`.

`SerializedFloat` is the serializable `cbrkit.typing.Float`,
so results keep the details of the values such as `AttributeValueSim` through JSON by their kind.
Every value with local details names them `similarities` and, if it aligns elements, their alignment `mapping`,
a graph adds `edge_similarities` and `edge_mapping` for its edges.
"""

from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from ..helpers import frozendict
from ..schema import Family, serializable
from ..schema._family import is_member
from ..typing import Sim
from .changes import Add, Rule, Update

__all__ = [
    "Alignment",
    "AttributeValueSim",
    "ChangeSim",
    "CritiqueSim",
    "GraphSim",
    "SequenceSim",
    "SerializedFloat",
]


type Alignment[V] = tuple[tuple[V | None, V | None], ...]
"""An alignment of two sequences as pairs of a query and a case element, None for a skipped one."""


def _detailed(cls: type[Sim]) -> Iterator[type[Sim]]:
    """The subclasses of a similarity value that serialize by their kind, recursively."""
    for subclass in cls.__subclasses__():
        if is_member(subclass):
            yield subclass

        yield from _detailed(subclass)


class _SimValues:
    """Validates and serializes a `Sim` as one of its subclasses with a `kind` field, by that kind."""

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        # subclasses outside this module may be defined later, so they are collected once a schema needs them
        family = Family("similarity value", tuple(dict.fromkeys(_detailed(Sim))))

        return family.__get_pydantic_core_schema__(source, handler)


type SerializedFloat = float | Annotated[Sim, _SimValues()]
"""The `cbrkit.typing.Float` of pydantic models, which keeps detailed similarities through JSON by their kind.

Type checkers treat both as the same type.
"""


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class AttributeValueSim[S: SerializedFloat](Sim):
    """Result of an attribute-value similarity computation with per-attribute scores.

    Args:
        similarities: The local similarities by attribute name.
    """

    similarities: frozendict[str, S]
    kind: Literal["attribute_value"] = field(default="attribute_value", repr=False)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class CritiqueSim(Sim):
    """Assessment of a case by a language model, see `cbrkit.sim.llm.Critique`.

    Args:
        issues: The problems that the model found.
        model: Name of the language model.
    """

    issues: tuple[str, ...] = ()
    model: str = ""
    kind: Literal["critique"] = field(default="critique", repr=False)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class SequenceSim[V, S: SerializedFloat](Sim):
    """Similarity of two sequences with optional local scores and alignment.

    Args:
        similarities: Local similarities of the aligned elements.
        mapping: Alignment of the elements as pairs of a query and a case element, None for a skipped one.
    """

    similarities: tuple[S, ...] | None = None
    # spelled out, since pydantic names a definition after a type alias in JSON Schemas
    mapping: tuple[tuple[V | None, V | None], ...] | None = None
    kind: Literal["sequence"] = field(default="sequence", repr=False)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class GraphSim[K: Hashable = str | int](Sim):
    """Result of a graph similarity computation with node and edge mappings.

    Nodes are the primary elements, so their details use the common names of all similarity values,
    while the edges add their own.

    Args:
        similarities: Similarities of the mapped query nodes as returned by the node similarity,
            so details such as per-attribute scores are kept.
        mapping: Case node keys by query node key.
        edge_similarities: Similarities of the mapped query edges.
        edge_mapping: Case edge keys by query edge key.
        exact: Whether the algorithm proved the mapping to be optimal.

    Unparametrized, the keys are strings or integers, which keeps its JSON Schema finite.
    """

    similarities: frozendict[K, SerializedFloat]
    mapping: frozendict[K, K]
    edge_similarities: frozendict[K, float]
    edge_mapping: frozendict[K, K]
    exact: bool = False
    kind: Literal["graph"] = field(default="graph", repr=False)


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class ChangeSim[K = str | int](Sim):
    """The change that stored a case in a retain run, see `cbrkit.retain.Retainer`.

    Its value is the similarity of the stored case that a duplicate is merged into, 1 for an added case.
    It keeps every field of the change but the value, so `ChangeSim.change` restores the change.

    Args:
        op: Whether the case was added or a stored case was updated.
        rule: The policy behind the change, None for a write of an application.
        reason: Why the change was made, for people.
        score: The similarity that triggered the change, such as that of a duplicate.
        related: Keys of the cases that the change refers to, such as the duplicate of a solution.

    Examples:
        >>> sim = ChangeSim.of(Update(key=1, value="b", reason="merged", score=0.9, related=[0]))
        >>> float(sim)
        0.9
        >>> sim.change(1, "b")
        Update(rule=None, reason='merged', score=0.9, related=(0,), op='update', key=1, value='b')
    """

    op: Literal["add", "update"]
    rule: Rule | None
    reason: str
    score: float | None = None
    related: tuple[K, ...] = ()
    kind: Literal["change"] = field(default="change", repr=False)

    @staticmethod
    def of[T](change: Add[T, Any] | Update[T, Any]) -> ChangeSim[T]:
        """The similarity of a case that a change stored."""
        return ChangeSim(
            value=1.0 if change.score is None else change.score,
            op=change.op,
            rule=change.rule,
            reason=change.reason,
            score=change.score,
            related=tuple(change.related),
        )

    def change[V](self, key: K, value: V) -> Add[K, V] | Update[K, V]:
        """The change that stored a case under its key."""
        kind = Add if self.op == "add" else Update

        return kind(
            key=key,
            value=value,
            rule=self.rule,
            reason=self.reason,
            score=self.score,
            related=self.related,
        )
