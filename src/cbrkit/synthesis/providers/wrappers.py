from collections.abc import Sequence
from dataclasses import dataclass

from ...helpers import (
    batchify_conversion,
    produce_sequence,
    unbatchify_conversion,
    unpack_value,
    unpack_values,
)
from ...typing import (
    AnyConversionFunc,
    BatchConversionFunc,
    ConversionFunc,
    MaybeSequence,
    Value,
)
from .model import ChatMessage, ChatPrompt


@dataclass(slots=True, frozen=True)
class conversation[P, R](ConversionFunc[P, R]):
    generation_func: AnyConversionFunc[ChatPrompt[P], Value[R]]
    conversion_func: ConversionFunc[R, P]
    chat_func: ConversionFunc[list[ChatMessage], P | None]

    def __call__(self, batch: P) -> R:
        func = unbatchify_conversion(self.generation_func)

        messages: list[ChatMessage] = [ChatMessage(role="user", content=batch)]
        last_assistant_message: R = unpack_value(func(ChatPrompt(batch, messages)))
        messages.append(
            ChatMessage(
                role="assistant",
                content=self.conversion_func(last_assistant_message),
            )
        )

        while next_batch := self.chat_func(messages):
            messages.append(ChatMessage(role="user", content=next_batch))
            last_assistant_message = unpack_value(
                func(ChatPrompt(next_batch, messages))
            )

            messages.append(
                ChatMessage(
                    role="assistant",
                    content=self.conversion_func(last_assistant_message),
                )
            )

        return last_assistant_message


@dataclass(slots=True, frozen=True)
class pipe[P, R](BatchConversionFunc[P, R]):
    generation_funcs: MaybeSequence[AnyConversionFunc[P, Value[R]]]
    conversion_func: ConversionFunc[R, P]

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        funcs = produce_sequence(self.generation_funcs)
        current_input = batches
        current_output: Sequence[R] = []

        for func in funcs:
            batch_func = batchify_conversion(func)
            current_output = unpack_values(batch_func(current_input))
            current_input = [self.conversion_func(output) for output in current_output]

        if not len(current_output) == len(batches):
            raise ValueError(
                "The number of outputs does not match the number of inputs, "
                "did you provie a generation function?"
            )

        return current_output
