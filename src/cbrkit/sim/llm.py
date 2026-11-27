"""Similarity judged by a language model of the host, see `Critique` and `Fit`, and the binding of language models.

Measures and adaptations, such as `Critique` and `cbrkit.adapt.llm.Rewrite`,
refer to a language model by its name only, a `LanguageModel`,
so a specification holds no provider settings or secrets.
A name is bound to a `cbrkit.typing.GenerationFunc` in one of two ways:

- The `providers` section of a `cbrkit.spec` sidecar maps it to a Pydantic AI model identifier,
  such as `anthropic:claude-sonnet-4-6`, which `cbrkit.spec.providers` turns into `PydanticAI`.
  This needs the `pydantic-ai` extra, and keys and endpoints come from the environment variables of the provider.
- The host passes its own function by name, to `cbrkit.system.System` or `cbrkit.spec.providers`, which takes precedence over the providers section and needs no extra.
  Any async callable with the signature of `cbrkit.typing.GenerationFunc` works.

Measures and adaptations are synchronous, so they wait for the asynchronous language model with `cbrkit.helpers.wait`,
which awaits it on the event loop of `cbrkit.cycle.run_async` from the worker thread of the run.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, override

from pydantic import BaseModel, Field
from pydantic_core import to_json

from .._lazy import lazy_exports
from ..helpers import wait
from ..model.similarity import CritiqueSim
from ..schema import HostFunction, serializable
from ..typing import BatchSimFunc, GenerationFunc, SimSeq

if TYPE_CHECKING:
    from ._pydantic_ai import PydanticAI

__all__ = [
    "Critique",
    "Fit",
    "Fitness",
    "Judgement",
    "LanguageModel",
    "PydanticAI",
    "prompt",
]


@dataclass(slots=True, frozen=True)
class LanguageModel(HostFunction[GenerationFunc]):
    """A language model of the host, serialized by its name only, see `cbrkit.schema.HostFunction`.

    The `models` of the providers section of `cbrkit.spec` name the language models.
    """

    async def __call__[T](
        self, prompts: Sequence[str], response_type: type[T], /
    ) -> Sequence[T]:
        """Generates a response of a type for every prompt.

        Raises:
            cbrkit.errors.Unbound: If no function is bound to the name.
        """
        func = self.bound()

        return await func(prompts, response_type) if prompts else []


def prompt(instructions: str, **values: object) -> str:
    """A prompt of instructions followed by values as JSON, the same for every call.

    Examples:
        >>> print(prompt("Compare.", query={"a": 1}))
        Compare.
        <BLANKLINE>
        query: {"a":1}
    """
    parts = [
        f"{name}: {to_json(value, fallback=str).decode()}"
        for name, value in values.items()
    ]

    return "\n\n".join([instructions, "\n".join(parts)])


def _respond[V, R](
    model: LanguageModel,
    instructions: str,
    batches: Sequence[tuple[V, V]],
    response_type: type[R],
) -> Sequence[R]:
    """The responses of a model to the instructions about every pair of a case and a query, waiting for them."""
    return wait(
        model(
            [prompt(instructions, query=query, case=case) for case, query in batches],
            response_type,
        )
    )


class Judgement(BaseModel):
    """The judgement of a language model about a case for a query.

    Args:
        score: How well the case solves the query, from 0 for not at all to 1 for perfectly.
        issues: The problems of the case, empty if there are none.
    """

    score: float = Field(ge=0, le=1)
    issues: list[str] = []


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Critique[V](BatchSimFunc[V, CritiqueSim]):
    """Assessment of a case for a query by a language model, typically of a solution in revise.

    The model judges every pair with the same prompt, the instructions followed by the query and the case as JSON,
    and its score and issues become a `cbrkit.model.CritiqueSim` that records the name of the model.

    Args:
        model: Name of a language model of the host, see the providers section of `cbrkit.spec`.
        instructions: What makes a case a good solution of the query.

    Examples:
        >>> async def judge(prompts, response_type):
        ...     return [response_type(score=0.5, issues=["too expensive"]) for _ in prompts]
        >>> critique = Critique(model=LanguageModel("judge", judge), instructions="Judge the offer.")
        >>> critique([({"price": 30}, {"price": 20})])
        [CritiqueSim(value=0.5, issues=('too expensive',), model='judge')]
    """

    model: LanguageModel
    instructions: str
    kind: Literal["critique"] = field(default="critique", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]], /) -> SimSeq[CritiqueSim]:
        judgements = _respond(self.model, self.instructions, batches, Judgement)

        return [
            CritiqueSim(
                value=judgement.score,
                issues=tuple(judgement.issues),
                model=self.model.name,
            )
            for judgement in judgements
        ]


class Fitness(BaseModel):
    """Whether a case fits a query, the probability of which is the similarity."""

    fits: float = Field(
        ge=0,
        le=1,
        description="Does the case fit the query as the instructions describe?",
    )


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Fit[V](BatchSimFunc[V, float]):
    """Similarity of a case to a query as the probability that it fits, judged by a model.

    Unlike `Critique`, the model writes nothing, so a decision model judges as well as a language model,
    the decision model with a calibrated probability, such as Pydantic AI's `system-one:<model>`.

    Args:
        model: Name of a model of the host, see the providers section of `cbrkit.spec`.
        instructions: What makes a case fit the query.

    Examples:
        >>> async def judge(prompts, response_type):
        ...     return [response_type(fits=0.75) for _ in prompts]
        >>> Fit(model=LanguageModel("judge", judge), instructions="Is the offer affordable?")([({"price": 30}, {"price": 20})])
        [0.75]
    """

    model: LanguageModel
    instructions: str
    kind: Literal["fit"] = field(default="fit", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]], /) -> SimSeq[float]:
        return [
            item.fits
            for item in _respond(self.model, self.instructions, batches, Fitness)
        ]


lazy_exports(__name__)
