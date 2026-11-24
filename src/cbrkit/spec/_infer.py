"""Deterministic profiling of case bases, heuristic decisions, and merging with existing decisions."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from functools import cache
from typing import Any, cast

from ..adapt.generic import Null
from ..helpers import format_number
from ..io import Source
from ..retain import Retainer
from ..reuse import Reuser
from ..sim import dates
from ..sim.aggregator import Aggregator
from ..sim.attribute_value import AttributeValue
from ..sim.collections.jaccard import Jaccard
from ..sim.collections.mapping import OptimalMapping
from ..sim.embed.core import Embedder, Embedding
from ..sim.embed.metrics import Cosine
from ..sim.generic import Equality
from ..sim.graphs import AnyElement, SameAttribute, SemanticEdge
from ..sim.graphs.lap import LAP
from ..sim.numbers import Linear
from ..sim.strings import Levenshtein, Ngram
from ..sim.taxonomy import SerializedTaxonomyNode, Taxonomy
from ..sim.wrappers import AttributeTable
from ..typing import AnySimFunc, Casebase
from ._config import Config, Provenance
from ._profile import Profile, fingerprint, profile
from .pointer import escape, get

__all__ = ["infer"]

ID_TOKENS = frozenset({"id", "uuid", "guid", "url", "uri", "href", "email"})
"""Name tokens marking identifiers, which say nothing about similarity."""
MAX_CATEGORIES = 20
MAX_SHORT_LENGTH = 40
BPMN_KEPT = frozenset({"type", "name", "lane", "pool", "condition"})
"""BPMN attributes with domain meaning, all others are technical."""


@cache
def _bpmn_taxonomy() -> SerializedTaxonomyNode:
    """Hierarchy of the BPMN element types of the nodes of `cbrkit.model.graph.from_bpmn`, derived from the BPMN schemas."""
    from ..model.graph._bpmn_types import BpmnHierarchy, bpmn_hierarchy

    def taxonomy(hierarchy: BpmnHierarchy) -> SerializedTaxonomyNode:
        name, children = hierarchy

        return SerializedTaxonomyNode(
            name=name,
            children=[taxonomy(child) if child[1] else child[0] for child in children],
        )

    return taxonomy(bpmn_hierarchy())


def _tokens(name: str) -> set[str]:
    return {
        token.lower() for token in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", name)
    }


@dataclass(slots=True, frozen=True)
class _Choice:
    """A measure for an element of the case base with the decisions taken for it."""

    measure: AnySimFunc[Any, Any]
    provenance: Provenance
    weight: float = 1.0
    nested: Mapping[str, Provenance] = field(default_factory=dict)
    """Decisions of nested measures by their pointer relative to `measure`."""
    attributes: Mapping[str, _Choice] | None = None
    """The choices of the attributes of an `AttributeValue` measure."""

    def pointers(self, prefix: str = "") -> dict[str, Provenance]:
        return {
            prefix: self.provenance,
            **{prefix + pointer: value for pointer, value in self.nested.items()},
        }


def _leaf(
    measure: AnySimFunc[Any, Any], reason: str, element: str, weight: float = 1.0
) -> _Choice:
    return _Choice(measure, Provenance(reason=reason, element=element), weight)


def _object(attributes: Mapping[str, _Choice], provenance: Provenance) -> _Choice:
    measure = AttributeValue(
        attributes={name: choice.measure for name, choice in attributes.items()},
        aggregator=Aggregator(
            weights={name: choice.weight for name, choice in attributes.items()}
        ),
        default=0.0,
        skip_missing_query=True,
    )
    nested = {
        pointer: value
        for name, choice in attributes.items()
        for pointer, value in choice.pointers(f"/attributes/{escape(name)}").items()
    }

    return _Choice(measure, provenance, nested=nested, attributes=attributes)


def _ignored(p: Profile, name: str) -> str | None:
    """The reason why a value is only compared for equality and weighs nothing, None if it is compared by similarity."""
    if p.type == "missing":
        return "no values"

    if p.type == "mixed":
        return "mixed value types"

    if _tokens(name) & ID_TOKENS or p.pattern in ("url", "email"):
        return "identifier or link"

    if p.type == "string" and p.pattern == "token" and p.distinct == p.count >= 10:
        return "unique codes without whitespace"

    return None


def _decide(p: Profile, name: str, embedder: str | None, element: str) -> _Choice:
    def leaf(
        measure: AnySimFunc[Any, Any], reason: str, weight: float = 1.0
    ) -> _Choice:
        return _leaf(measure, reason, element, weight)

    if reason := _ignored(p, name):
        return leaf(Equality(), reason, 0.0)

    match p.type:
        case "bool":
            return leaf(Equality(), "boolean")
        case "number" | "date" | "duration":
            low, high = p.low or 0.0, p.high or 0.0
            span = high - low if high > low else abs(high) or 1.0
            reason = f"distance {format_number(span)} spans the 5th to 95th percentile"

            if p.type == "number":
                return leaf(Linear(max_distance=span), f"number, {reason}")

            return leaf(
                dates.DateLinear(max_distance=timedelta(days=span)),
                f"{p.type}, {reason} in days",
            )
        case "string" if (
            p.distinct <= MAX_CATEGORIES and 2 * p.distinct <= p.count
        ) or p.distinct <= 0.05 * p.count:
            return leaf(
                Equality(case_sensitive=False, normalize_whitespace=True),
                f"categorical text with {p.distinct} value{'' if p.distinct == 1 else 's'}",
            )
        case "string" if (p.mean_length or 0.0) <= MAX_SHORT_LENGTH:
            return leaf(Levenshtein(case_sensitive=False), "short text")
        case "string" if embedder is not None:
            return leaf(
                Embedding(embedder=Embedder(embedder), metric=Cosine()), "long text"
            )
        case "string":
            return leaf(Ngram(n=3, case_sensitive=False), "long text without embedder")
        case "list" if p.element is not None and p.element.type == "object":
            element_choice = _decide(p.element, name, embedder, f"{element}/element")

            return _Choice(
                OptimalMapping(element_measure=element_choice.measure),
                Provenance(reason="list of objects", element=element),
                nested=element_choice.pointers("/element_measure"),
            )
        case "list":
            return leaf(Jaccard(), "list of values")
        case "object":
            return _object(
                {
                    key: _decide(
                        value, key, embedder, f"{element}/attributes/{escape(key)}"
                    )
                    for key, value in p.attributes.items()
                },
                Provenance(reason="object", element=element),
            )
        case _:
            return _graph_decision(p, embedder, element)


def _adjust(choice: _Choice, updates: Mapping[str, _Choice]) -> _Choice:
    """Replaces attributes of an `AttributeValue` choice."""
    if choice.attributes is None:
        return choice

    return replace(
        _object({**choice.attributes, **updates}, choice.provenance),
        weight=choice.weight,
    )


def _excluded(choice: _Choice, reason: str) -> _Choice:
    return replace(
        choice,
        weight=0.0,
        provenance=choice.provenance.model_copy(update={"reason": reason}),
    )


def _bpmn(choice: _Choice, typed: bool) -> _Choice:
    if choice.attributes is None:
        return choice

    updates = {
        name: _excluded(value, "technical BPMN attribute")
        for name, value in choice.attributes.items()
        if name not in BPMN_KEPT
    }

    if typed and "type" in choice.attributes:
        updates["type"] = _leaf(
            Taxonomy(taxonomy=_bpmn_taxonomy(), default=0.0),
            "BPMN element type hierarchy",
            choice.attributes["type"].provenance.element,
        )

    return _adjust(choice, updates)


def _is_bpmn(p: Profile) -> bool:
    """Whether the node types of a graph profile are BPMN elements."""
    if p.node_type != "type":
        return False

    from ..model.graph._bpmn_types import BPMN_NODES, bpmn_class

    return all(issubclass(bpmn_class(key), BPMN_NODES) for key in p.node_types)


def _graph_decision(p: Profile, embedder: str | None, element: str) -> _Choice:
    node_type = p.node_type
    bpmn = _is_bpmn(p)
    node = _decide(
        p.node or Profile(type="missing"), "node", embedder, f"{element}/node"
    )
    node_types: dict[str, _Choice] = {}

    for key, value in p.node_types.items():
        choice = _decide(value, key, embedder, f"{element}/node_types/{escape(key)}")

        if (
            node_type is not None
            and choice.attributes
            and node_type in choice.attributes
        ):
            choice = _adjust(
                choice,
                {
                    node_type: _excluded(
                        choice.attributes[node_type], "constant within a node type"
                    )
                },
            )

        node_types[key] = _bpmn(choice, False) if bpmn else choice

    edge = (
        _decide(p.edge, "edge", embedder, f"{element}/edge")
        if p.edge is not None and p.edge.type != "missing"
        else None
    )

    if bpmn:
        node = _bpmn(node, True)
        edge = edge and _bpmn(edge, False)

    nested: dict[str, Provenance] = {}

    if node_type is not None and node_types:
        node_measure: AnySimFunc[Any, Any] = AttributeTable(
            attribute=node_type,
            entries={key: value.measure for key, value in node_types.items()},
            fallback=node.measure,
        )
        nested |= node.pointers("/node_measure/fallback")

        for key, value in node_types.items():
            nested |= value.pointers(f"/node_measure/entries/{escape(key)}")
    else:
        node_measure = node.measure
        nested |= node.pointers("/node_measure")

    if edge is not None:
        nested |= edge.pointers("/edge_measure/value_measure")

    return _Choice(
        LAP(
            node_measure=node_measure,
            edge_measure=SemanticEdge(
                value_measure=edge.measure if edge is not None else None
            ),
            node_matcher=SameAttribute(attribute=node_type)
            if node_type is not None and not bpmn
            else AnyElement(),
        ),
        Provenance(
            reason="BPMN graph mapped with LAP" if bpmn else "graph mapped with LAP",
            element=element,
        ),
        nested=nested,
    )


def infer(
    casebase: Casebase[Any, Any],
    source: Source | None = None,
    saved: Config | None = None,
    *,
    embedder: str | None = None,
    samples: int = 20,
    type_attribute: str = "type",
) -> Config:
    """The configuration of a case base: heuristic decisions for every element, or those of the saved one.

    Besides the measure, it fills the sections that need no decision about the domain:
    retrieve with its default, reuse with a null adaptation that proposes the most similar case as it is,
    and retain, which stores every solution, if the source is writable.
    Revise and maintain need thresholds that only the domain decides, so they stay off.

    A saved configuration is returned as it is while the shape of the cases is unchanged,
    otherwise it gets the heuristic decisions for the elements it lacks.
    Its existing decisions are never rewritten, whatever their source,
    and measures whose kind changed keep the saved one.
    Elements that no longer occur are kept and marked as `absent`, which does not
    change retrieval since missing values are skipped or get the default similarity.
    So a host knows that a result equal to `saved` needs no new sidecar.

    Args:
        casebase: The cases, plain Python records or cbrkit graphs.
        source: How the case base is stored, needed without a saved configuration, whose source is kept.
        saved: The configuration of the sidecar, if there is one.
        embedder: Name of the host embedder for long texts, n-grams are used if None.
        samples: Number of distinct sample values kept per element while profiling.
        type_attribute: Attribute of the node values of graphs that holds their type,
            whose nodes are compared per type.

    Raises:
        ValueError: If neither a source nor a saved configuration is given.

    Examples:
        >>> config = infer({1: {"id": 1, "price": 10.0}, 2: {"id": 2, "price": 30.0}}, Source(format="csv"))
        >>> config.measure.attributes["price"], config.measure.aggregator.weights["id"]
        (Linear(max_distance=18.0, min_distance=0.0), 0.0)
        >>> config.provenance["/attributes/price"].reason
        'number, distance 18 spans the 5th to 95th percentile'
        >>> infer({1: {"id": 3, "price": 20.0}, 2: {"id": 4, "price": 40.0}}, saved=config) == config
        True
        >>> merged = infer({1: {"price": 1.0, "fuel": "gas"}}, saved=config)
        >>> list(merged.measure.attributes), merged.provenance["/attributes/id"].absent
        (['id', 'price', 'fuel'], True)
    """
    if saved is not None:
        source = saved.source
    elif source is None:
        raise ValueError("A case base without a saved configuration needs its source")

    p = profile(casebase.values(), samples, type_attribute)
    shape = fingerprint(p)

    if saved is not None and saved.fingerprint == shape:
        return saved

    choice = _decide(p, "", embedder, "")
    inferred = Config(
        source=source,
        fingerprint=shape,
        measure=choice.measure,
        provenance=choice.pointers(),
        reuse=Reuser(adaptation=Null()),
        retain=Retainer() if source.writable else None,
    )

    return inferred if saved is None else _merge(saved, inferred)


def _add(old: Any, new: Any) -> Any:
    """Adds what the new serialized measure has and the old one lacks, keeping the old values."""
    if (
        isinstance(old, dict)
        and isinstance(new, dict)
        and old.get("kind") == new.get("kind")
    ):
        merged = cast(dict[str, Any], old)

        return {
            **{key: _add(value, new.get(key)) for key, value in merged.items()},
            **{key: value for key, value in new.items() if key not in merged},
        }

    return old


def _merge(existing: Config, inferred: Config) -> Config:
    """Adds the elements of an inferred configuration that the existing one lacks, see `infer`."""
    old, new = existing.model_dump(mode="json"), inferred.model_dump(mode="json")
    measure = _add(old["measure"], new["measure"])
    decisions = {
        pointer: {**value, "absent": pointer not in new["provenance"]}
        for pointer, value in old["provenance"].items()
    } | {
        pointer: value
        for pointer, value in new["provenance"].items()
        if pointer not in old["provenance"] and get(measure, pointer) is not None
    }

    return type(existing).model_validate(
        old
        | {
            "fingerprint": new["fingerprint"],
            "measure": measure,
            "provenance": decisions,
        }
    )
