from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from ..genai._wrappers import transpose
from ..helpers import identity, unbatchify_generation
from ..typing import (
    AdaptationMapFunc,
    AnyGenerationFunc,
    BatchGenerationFunc,
    Casebase,
    PromptFunc,
)

__all__ = [
    "build",
    "PydanticModel",
    "unpack_pydantic_model",
]


class PydanticModel[K, V](BaseModel):
    casebase: Mapping[K, V]


def _unpack_pydantic_model[K, V](obj: PydanticModel[K, V]) -> Mapping[K, V]:
    return obj.casebase


def unpack_pydantic_model[P, K, V](
    generation_func: AnyGenerationFunc[P, PydanticModel[K, V]],
) -> BatchGenerationFunc[P, Mapping[K, V]]:
    return transpose(
        generation_func,
        prompt_conversion_func=identity,
        response_conversion_func=_unpack_pydantic_model,
    )


@dataclass(slots=True, frozen=True)
class build[P, K, V: BaseModel](AdaptationMapFunc[K, V]):
    generation_func: AnyGenerationFunc[P, Mapping[K, V]]
    prompt_func: PromptFunc[P, K, V, Any]

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V,
    ) -> Casebase[K, V]:
        generation_func = unbatchify_generation(self.generation_func)
        return generation_func(self.prompt_func(casebase, query, None))
