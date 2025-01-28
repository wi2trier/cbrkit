from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ...helpers import (
    batchify_conversion,
    produce_sequence,
    unbatchify_conversion,
)
from ...typing import (
    AnyConversionFunc,
    BatchConversionFunc,
    ConversionFunc,
    MaybeSequence,
)
from .model import ChatMessage, ChatPrompt


@dataclass(slots=True, frozen=True)
class conversation[R](ConversionFunc[Iterable[str], R]):
    generation_func: AnyConversionFunc[ChatPrompt[str], R]
    conversion_func: ConversionFunc[R, str]

    def __call__(self, prompts: Iterable[str]) -> R:
        func = unbatchify_conversion(self.generation_func)

        messages: list[ChatMessage] = []
        last_assistant_message: R | None = None

        for prompt in prompts:
            messages.append(ChatMessage(role="user", content=prompt))
            last_assistant_message = func(ChatPrompt(prompt, messages))

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=self.conversion_func(last_assistant_message),
                )
            )

        if last_assistant_message is None:
            raise ValueError("The last assistant message is empty")

        return last_assistant_message


@dataclass(slots=True, frozen=True)
class pipe[P, R](BatchConversionFunc[P, R]):
    generation_funcs: MaybeSequence[AnyConversionFunc[P, R]]
    conversion_func: ConversionFunc[R, P]

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        funcs = produce_sequence(self.generation_funcs)
        current_input = batches
        current_output: Sequence[R] = []

        for func in funcs:
            wrapped_func = batchify_conversion(func)
            current_output = wrapped_func(current_input)
            current_input = [self.conversion_func(output) for output in current_output]

        if not len(current_output) == len(batches):
            raise ValueError(
                "The number of outputs does not match the number of inputs, "
                "did you provie a generation function?"
            )

        return current_output
