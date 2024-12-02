from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..genai._wrappers import transpose
from ..helpers import batchify_generation, identity
from ..typing import (
    AnyGenerationFunc,
    BatchGenerationFunc,
    Casebase,
    PromptFunc,
    RetrieverFunc,
)

__all__ = [
    "build",
    "PydanticModel",
    "unpack_pydantic_model",
]


class PydanticModel[K](BaseModel):
    similarities: Mapping[K, float]


def _unpack_pydantic_model[K](obj: PydanticModel[K]) -> Mapping[K, float]:
    return obj.similarities


def unpack_pydantic_model[P, K](
    generation_func: AnyGenerationFunc[P, PydanticModel[K]],
) -> BatchGenerationFunc[P, Mapping[K, float]]:
    return transpose(
        generation_func,
        prompt_conversion_func=identity,
        response_conversion_func=_unpack_pydantic_model,
    )


@dataclass(slots=True, frozen=True)
class build[P, K, V](RetrieverFunc[K, V, float]):
    generation_func: AnyGenerationFunc[P, Mapping[K, float]]
    prompt_func: PromptFunc[P, K, V, Any]

    def __call__(
        self,
        batches: Sequence[tuple[Casebase[K, V], V]],
    ) -> Sequence[Casebase[K, float]]:
        generation_func = batchify_generation(self.generation_func)
        return generation_func([self.prompt_func(*batch, None) for batch in batches])
