"""Adaptation by a language model of the host, see `Rewrite`."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, override

from pydantic import JsonValue

from ..helpers import replace_attributes, wait
from ..model import solution, with_solution
from ..schema import serializable
from ..sim.llm import LanguageModel, prompt
from ..typing import BatchAdaptationFunc

__all__ = ["Rewrite"]


@serializable
@dataclass(slots=True, frozen=True, kw_only=True)
class Rewrite[V](BatchAdaptationFunc[V]):
    """Adapts the solution of a case to a query with a language model.

    The model gets the same prompt for every pair: the instructions followed by the context, the query, and the case as JSON.
    With `attributes`, or for a mapping, it answers with the new values of the attributes it may change by name,
    and all others keep the values of the solution.
    Otherwise it answers with the whole solution as structured output of its type, such as a dataclass or a pydantic model.
    A split `cbrkit.model.Case` keeps its problem, see `cbrkit.reuse.Reuser`.
    The adaptation is synchronous and waits for the language model with `cbrkit.helpers.wait`.

    Args:
        model: Name of a language model of the host, see the providers section of `cbrkit.spec`.
        instructions: How to adapt a solution to the query.
        attributes: Attributes that the model may change, all attributes of a mapping or the whole solution if None.
        context: Further values the model needs, such as the tools that a plan may use.

    Examples:
        >>> from cbrkit.sim.llm import LanguageModel
        >>> async def writer(prompts, response_type):
        ...     return [{"price": 20, "color": "red"} for _ in prompts]
        >>> rewrite = Rewrite(model=LanguageModel("writer", writer), instructions="Match the price.", attributes=["price"])
        >>> rewrite([({"price": 30, "color": "blue"}, {"price": 20})])
        [{'price': 20, 'color': 'blue'}]
        >>> from cbrkit.model import Case
        >>> async def planner(prompts, response_type):
        ...     return [response_type("walk") for _ in prompts]
        >>> rewrite = Rewrite(model=LanguageModel("planner", planner), instructions="Plan the trip.")
        >>> rewrite([(Case(problem="park", solution="drive"), Case(problem="beach"))])
        [Case(problem='park', solution='walk')]
    """

    model: LanguageModel
    instructions: str
    attributes: Sequence[str] | None = None
    context: Mapping[str, JsonValue] = field(default_factory=dict)
    kind: Literal["rewrite"] = field(default="rewrite", repr=False)

    @override
    def __call__(self, batches: Sequence[tuple[V, V]], /) -> Sequence[V]:
        if not batches:
            return []

        solutions = [solution(case) for case, _ in batches]
        partial = self.attributes is not None or all(
            isinstance(value, Mapping) for value in solutions
        )
        response_type: type[Any] = (
            dict[str, JsonValue] if partial else type(solutions[0])
        )
        responses = wait(
            self.model(
                [
                    prompt(self.instructions, **self.context, query=query, case=case)
                    for case, query in batches
                ],
                response_type,
            )
        )

        return [
            with_solution(case, self._apply(value, response) if partial else response)
            for (case, _), value, response in zip(
                batches, solutions, responses, strict=True
            )
        ]

    def _apply(self, value: Any, response: Mapping[str, Any]) -> Any:
        allowed = self.attributes if self.attributes is not None else list(value)

        return replace_attributes(
            value, {name: item for name, item in response.items() if name in allowed}
        )
