"""Refinement of decisions by a language model with validation and fallback, and by a user.

The model only chooses deterministic leaf measures (`leaves`), those nesting no other measure and using no language model,
and attribute weights,
for decisions that exist and were not taken by a user.
`refine` profiles the cases, prompts a `cbrkit.sim.llm.LanguageModel` for a response model that lists the editable paths,
and builds and runs every proposal on sample values of the case base before it refines the configuration,
so rejected proposals keep the previous decision.
The patches apply to the measures themselves, so custom measures are refined like built-in ones.
A configuration edited by a user goes through `mark_user`,
which records every changed decision as taken by the user, so later refinements keep it.
"""

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from functools import cache
from itertools import pairwise
from typing import Annotated, Any

import orjson
from pydantic import Field, GetPydanticSchema, JsonValue

from ..helpers import batchify, to_thread
from ..schema import Family, Model, resolve
from ..sim import MEASURES
from ..sim.aggregator import Aggregator
from ..sim.attribute_value import AttributeValue
from ..sim.llm import LanguageModel
from ..typing import AnySimFunc, Casebase
from ._config import Config, Embedders, Provenance, _build, _Measure
from ._profile import Profile, profile
from .pointer import escape, get, parent, put

__all__ = ["Rejection", "leaves", "mark_user", "refine"]

INSTRUCTIONS = """\
You configure the similarity measures of a case-based retrieval system.
Every decision below is identified by the JSON pointer of its measure and compares the values of one element of the case base.
For each decision that the domain suggests to improve, return a patch with a new measure from the given schema, a new weight, or both, and a short reason.
Weights express the relative importance of an attribute, zero excludes it.
Only patch the listed paths, leave decisions out that are fine, and never invent attributes.
Tables and taxonomies must contain every observed value, equal values always have similarity 1.
Numeric parameters are distances in the units of the values, durations are ISO 8601 such as P30D.
"""


def leaves() -> Family:
    """The deterministic leaf measures, which nest no other measure and use no language model.

    A language model chooses from them in `refine`, and editors offer them to users by their kinds.
    Measures added with `cbrkit.schema.using` are included.

    Examples:
        >>> {"linear", "levenshtein"} <= set(leaves().by_kind)
        True
        >>> {"attribute_value", "critique"} & set(leaves().by_kind)
        set()
    """
    return _leaves(resolve(MEASURES))


@cache
def _leaves(measures: Family) -> Family:
    return measures.leaves().excluding(LanguageModel)


class _Patch(Model):
    """Proposed change of one decision.

    Args:
        path: JSON pointer of the decision.
        measure: The new measure.
        weight: The new weight of an attribute.
        reason: Short justification.
    """

    path: str
    measure: Annotated[AnySimFunc[Any, Any], leaves()] | None = None
    weight: Annotated[float, Field(ge=0)] | None = None
    reason: str


class _Refinement(Model):
    """Response of the language model.

    Args:
        patches: The proposed changes.
    """

    patches: Sequence[_Patch]


class Rejection(Model):
    """A patch that was not applied.

    Args:
        path: JSON pointer of the decision.
        reason: Why the patch was rejected.
    """

    path: str
    reason: str


def _aggregator(tree: Any, pointer: str) -> tuple[str, Aggregator[str], str] | None:
    """The pointer and aggregator of the attribute-value measure holding the attribute at a pointer, and its name."""
    if (found := parent(pointer)) is None:
        return None

    head, name = found

    if (container := parent(head)) is None or container[1] != "attributes":
        return None

    measure = get(tree, container[0])

    if not isinstance(measure, AttributeValue):
        return None

    return f"{container[0]}/aggregator", measure.aggregator, name


def _weight(tree: Any, pointer: str) -> float | None:
    """The weight of the attribute at a pointer, None if it is no attribute."""
    if (found := _aggregator(tree, pointer)) is None:
        return None

    _, aggregator, name = found

    return aggregator.weight(name)


def _json(measure: Any) -> Any:
    return _Measure(measure).model_dump(mode="json")


def _editable(config: Config) -> list[str]:
    return [
        pointer
        for pointer, decision in config.provenance.items()
        if decision.source != "user"
    ]


def _response_model(config: Config) -> type[_Refinement]:
    """Response model whose JSON schema lists the paths the model may patch."""
    paths: list[JsonValue] = [*_editable(config)]

    class ConfigPatch(_Patch):
        path: Annotated[str, Field(json_schema_extra={"enum": paths})]
        # the choices at call time, including measures added by `cbrkit.schema.using`
        measure: Annotated[AnySimFunc[Any, Any], leaves()] | None = None

    class ConfigRefinement(_Refinement):
        # validated as the patches above, which keeps the type of the field
        patches: Annotated[
            Sequence[_Patch],
            GetPydanticSchema(lambda _, handler: handler(Sequence[ConfigPatch])),
        ]

    return ConfigRefinement


def _profile(p: Profile, pointer: str) -> Profile | None:
    """The profile of an element by its pointer, None if there is none."""
    found = get(p, pointer)

    return found if isinstance(found, Profile) else None


def _summary(p: Profile | None, samples: int) -> dict[str, Any]:
    if p is None:
        return {}

    def shorten(value: Any) -> Any:
        return value[:80] if isinstance(value, str) else value

    fields = p.model_dump(
        mode="json",
        include={
            "type",
            "count",
            "missing",
            "distinct",
            "low",
            "high",
            "mean_length",
            "pattern",
        },
        exclude_none=True,
    )

    return {
        **fields,
        "samples": [
            shorten(value) for value in p.model_dump(mode="json")["samples"][:samples]
        ]
        if p.type not in ("object", "graph")
        else [],
    }


def _prompt(config: Config, stats: Profile, domain: str | None, samples: int) -> str:
    """Builds the prompt for a language model.

    Args:
        config: The current configuration.
        stats: Profile of the case base.
        domain: Optional description of the domain and the purpose of the retrieval.
        samples: Maximum number of sample values per element.
    """
    tree = config.measure
    choices = leaves().members
    editable = set(_editable(config))
    entries: list[dict[str, Any]] = []

    for pointer, decision in config.provenance.items():
        measure = get(tree, pointer)
        weight = _weight(tree, pointer)
        entries.append(
            {
                "path": pointer,
                "locked": pointer not in editable,
                "measure": _json(measure)
                if type(measure) in choices
                else {"kind": getattr(measure, "kind", type(measure).__name__)},
                **({"weight": weight} if weight is not None else {}),
                "data": _summary(_profile(stats, decision.element), samples),
            }
        )

    sections = [INSTRUCTIONS]

    if domain:
        sections.append(f"Domain:\n{domain}\n")

    sections.append(
        "Decisions (locked ones are fixed, composite ones only accept weights):\n"
        + "\n".join(orjson.dumps(entry).decode() for entry in entries)
    )

    return "\n".join(sections)


def _check(
    measure: AnySimFunc[Any, Any], values: Sequence[Any], embedders: Embedders
) -> str | None:
    """Builds a measure and runs it on sample pairs, returns the problem if any."""
    if not values:
        return "no sample values to verify the measure"

    pairs = [(value, value) for value in values] + list(pairwise(values))

    try:
        sims = [float(sim) for sim in batchify(_build(measure, embedders))(pairs)]
    except Exception as error:  # noqa: BLE001, any failure rejects the measure
        return f"measure fails on the data: {error!r}"

    if any(not 0.0 <= sim <= 1.0 or math.isnan(sim) for sim in sims):
        return "similarities outside of [0, 1]"

    if any(not math.isclose(sim, 1.0, abs_tol=1e-6) for sim in sims[: len(values)]):
        return "equal values are not fully similar"

    return None


def _apply(
    config: Config,
    refinement: _Refinement,
    stats: Profile,
    model: str,
    embedders: Embedders,
) -> tuple[Config, list[Rejection]]:
    """Validates the patches of a language model and returns the configuration refined by the valid ones."""
    tree = config.measure
    choices = leaves().members
    decisions = dict(config.provenance)
    rejections: list[Rejection] = []
    seen: set[str] = set()

    for patch in refinement.patches:
        decision = decisions.get(patch.path)
        problem: str | None = None

        if decision is None:
            rejections.append(Rejection(path=patch.path, reason="unknown path"))
            continue

        if patch.path in seen:
            problem = "duplicate patch"
        elif decision.source == "user":
            problem = "decided by a user"
        elif patch.measure is None and patch.weight is None:
            problem = "neither measure nor weight"
        elif patch.weight is not None and _weight(tree, patch.path) is None:
            problem = "only attributes have weights"
        elif patch.measure is not None and type(get(tree, patch.path)) not in choices:
            problem = "only deterministic leaf measures are replaced"
        elif patch.measure is not None:
            p = _profile(stats, decision.element)
            problem = _check(patch.measure, p.samples if p else [], embedders)

        if problem is not None:
            rejections.append(Rejection(path=patch.path, reason=problem))
            continue

        seen.add(patch.path)
        decisions[patch.path] = Provenance(
            source="llm", model=model, reason=patch.reason, element=decision.element
        )

        if patch.measure is not None:
            tree = put(tree, patch.path, patch.measure)

        if (
            patch.weight is not None
            and (found := _aggregator(tree, patch.path)) is not None
        ):
            at, aggregator, name = found
            weights = {**aggregator.weights, name: patch.weight}
            tree = put(tree, at, replace(aggregator, weights=weights))

    refined = config.model_copy(update={"measure": tree, "provenance": decisions})

    return refined, rejections


async def refine(
    config: Config,
    casebase: Casebase[Any, Any],
    model: LanguageModel,
    *,
    domain: str | None = None,
    embedders: Embedders | None = None,
    samples: int = 10,
    type_attribute: str = "type",
) -> tuple[Config, list[Rejection]]:
    """Refines the decisions of a configuration that no user took by a language model.

    The model sees statistics and sample values of the cases and proposes patches.
    A patch is rejected if its path is unknown or decided by a user, if it replaces
    a composite measure, if it sets the weight of a measure that is not an attribute,
    or if its measure fails on the sample values:
    errors, similarities outside of [0, 1], or equal values that are not fully similar.
    Rejected patches keep the previous decision.

    Args:
        config: The current configuration.
        casebase: The cases, profiled for the prompt and the sample values.
        model: The language model bound to a function, such as `cbrkit.sim.llm.PydanticAI` or any `cbrkit.typing.GenerationFunc`
            that returns the given response type, such as the output type of a Pydantic AI agent.
            Its name is recorded in the decisions.
        domain: Optional description of the domain and the purpose of the retrieval.
        embedders: Embedding functions to verify embedding measures.
        samples: Most sample values per element that the model sees.
        type_attribute: Attribute of node values that holds their type, as passed to `infer`.

    Raises:
        cbrkit.errors.Unbound: If the language model is bound to no function.

    Returns:
        The refined configuration and the rejected patches.
    """
    stats = await to_thread(profile, casebase.values(), type_attribute=type_attribute)
    (refinement,) = await model(
        [_prompt(config, stats, domain, samples)], _response_model(config)
    )

    return await to_thread(
        _apply, config, refinement, stats, model.name, embedders or {}
    )


def _own(tree: Any, pointer: str, pointers: Collection[str]) -> Any:
    """The measure at a pointer without the measures of other decisions and the weights of attributes below it."""

    def strip(node: Any, at: str) -> Any:
        if at != pointer and at in pointers:
            return None

        if isinstance(node, Mapping):
            return {
                key: strip(value, f"{at}/{escape(key)}") for key, value in node.items()
            }

        if is_dataclass(node):
            return type(node), {
                f.name: strip(getattr(node, f.name), f"{at}/{escape(f.name)}")
                for f in fields(node)
                if not (isinstance(node, Aggregator) and f.name == "weights")
            }

        return node

    return strip(get(tree, pointer), pointer)


def mark_user(previous: Config, edited: Config) -> Config:
    """Returns an edited configuration whose changed decisions are taken by a user.

    A decision changed if its own measure, without the measures of the decisions below it, or its weight differs,
    and unchanged decisions keep their previous provenance,
    so hosts compute the provenance of an edit instead of trusting their clients.

    Args:
        previous: The configuration before the edit.
        edited: The configuration edited by a user.

    Examples:
        >>> from cbrkit.io import Source
        >>> from cbrkit.spec import infer
        >>> previous = infer({1: {"a": 1.0, "b": "x"}, 2: {"a": 3.0, "b": "y"}}, Source(format="json"))
        >>> data = previous.model_dump(mode="json")
        >>> data["measure"]["aggregator"]["weights"]["b"] = 2.0
        >>> edited = mark_user(previous, Config.model_validate(data))
        >>> {pointer: decision.source for pointer, decision in edited.provenance.items()}
        {'': 'heuristic', '/attributes/a': 'heuristic', '/attributes/b': 'user'}
    """
    old, new = previous.measure, edited.measure
    pointers = set(edited.provenance)
    decisions = {
        pointer: Provenance(source="user", reason="edited", element=decision.element)
        if _own(old, pointer, pointers) != _own(new, pointer, pointers)
        or _weight(old, pointer) != _weight(new, pointer)
        else previous.provenance.get(pointer, decision)
        for pointer, decision in edited.provenance.items()
    }

    return edited.model_copy(update={"provenance": decisions})
