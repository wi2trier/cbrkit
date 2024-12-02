from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..helpers import (
    batchify_generation,
    unbatchify_generation,
)
from ..typing import (
    AnyGenerationFunc,
    BatchGenerationFunc,
    ConversionFunc,
    GenerationFunc,
)
from ._model import ChatMessage, ChatPrompt


@dataclass(slots=True, frozen=True)
class conversation[R](GenerationFunc[Iterable[str], R]):
    generation_func: AnyGenerationFunc[ChatPrompt[str], R]
    conversion_func: ConversionFunc[R, str]

    def __call__(self, prompts: Iterable[str]) -> R:
        func = unbatchify_generation(self.generation_func)

        messages: list[ChatMessage] = []
        last_assistant_message: R | None = None

        for prompt in prompts:
            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )
            last_assistant_message = func(ChatPrompt(prompt, messages))

            messages.append(
                {
                    "role": "assistant",
                    "content": self.conversion_func(last_assistant_message),
                }
            )

        if last_assistant_message is None:
            raise ValueError("The last assistant message is empty")

        return last_assistant_message


@dataclass(slots=True, frozen=True)
class transpose[P1, P2, R1, R2](BatchGenerationFunc[P1, R1]):
    generation_func: AnyGenerationFunc[P2, R2]
    prompt_conversion_func: ConversionFunc[P1, P2]
    response_conversion_func: ConversionFunc[R2, R1]

    def __call__(self, batches: Sequence[P1]) -> Sequence[R1]:
        func = batchify_generation(self.generation_func)
        responses = func([self.prompt_conversion_func(batch) for batch in batches])
        return [self.response_conversion_func(batch) for batch in responses]


@dataclass(slots=True, frozen=True)
class pipe[P, R](BatchGenerationFunc[P, R]):
    generation_funcs: Sequence[AnyGenerationFunc[P, R]] | AnyGenerationFunc[P, R]
    conversion_func: ConversionFunc[R, P]

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        funcs = (
            self.generation_funcs
            if isinstance(self.generation_funcs, Sequence)
            else [self.generation_funcs]
        )

        current_input = batches
        current_output: Sequence[R] = []

        for func in funcs:
            wrapped_func = batchify_generation(func)
            current_output = wrapped_func(current_input)
            current_input = [self.conversion_func(output) for output in current_output]

        if not len(current_output) == len(batches):
            raise ValueError(
                "The number of outputs does not match the number of inputs, "
                "did you provie a generation function?"
            )

        return current_output
