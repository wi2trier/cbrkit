from collections.abc import Sequence
from dataclasses import dataclass

from ..helpers import batchify_generation
from ..typing import AnyGenerationFunc, Casebase, Float, PromptFunc, RagFunc, SimMap
from ._apply import apply_batches, apply_result
from ._model import QueryResultStep, Result, ResultStep

__all__ = [
    "apply_batches",
    "apply_result",
    "build",
    "QueryResultStep",
    "ResultStep",
    "Result",
]


@dataclass(slots=True, frozen=True)
class build[P, R, K, V, S: Float](RagFunc[R, K, V, S]):
    generation_func: AnyGenerationFunc[P, R]
    prompt_func: PromptFunc[P, K, V, S]

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S]]]
    ) -> Sequence[R]:
        func = batchify_generation(self.generation_func)

        return func([self.prompt_func(*batch) for batch in batches])
