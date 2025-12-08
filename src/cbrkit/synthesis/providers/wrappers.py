from collections.abc import Sequence
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


@dataclass(slots=True, frozen=True)
class conversation[P, R](ConversionFunc[Sequence[P], R]):
    generation_func: AnyConversionFunc[Sequence[P], R]
    conversion_func: ConversionFunc[R, Sequence[P] | None]

    def __call__(self, batch: Sequence[P]) -> R:
        func = unbatchify_conversion(self.generation_func)
        result = func(batch)

        while next_batch := self.conversion_func(result):
            result = func(next_batch)

        return result


@dataclass(slots=True, frozen=True)
class pipe[P, R](BatchConversionFunc[P, R]):
    generation_funcs: MaybeSequence[AnyConversionFunc[P, R]]
    conversion_func: ConversionFunc[R, P]

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        funcs = produce_sequence(self.generation_funcs)
        current_input = batches
        current_output: Sequence[R] = []

        for func in funcs:
            batch_func = batchify_conversion(func)
            current_output = batch_func(current_input)
            current_input = [self.conversion_func(output) for output in current_output]

        if not len(current_output) == len(batches):
            raise ValueError(
                "The number of outputs does not match the number of inputs, "
                "did you provie a generation function?"
            )

        return current_output
