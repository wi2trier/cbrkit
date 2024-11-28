from collections.abc import Callable, Sequence
from dataclasses import dataclass
from textwrap import dedent

import orjson
from pydantic import BaseModel

from cbrkit.helpers import GenerationSeqWrapper, similarities2ranking, unpack_sim

from ..typing import AnyGenerationFunc, Casebase, Float, RagFunc, SimMap

__all__ = []


def default_prompt_template[K, V, S: Float](
    prompt: str,
    casebase: Casebase[K, V],
    query: V,
    similarities: SimMap[K, S],
) -> str:
    ranking: list[K] = similarities2ranking(similarities)

    result = dedent(f"""
        {prompt}

        ## Query

        ```json
        {str(orjson.dumps(query))}
        ```

        ## Retrieved Cases
    """)

    for rank, key in enumerate(ranking, start=1):
        result += dedent(f"""
            ### {rank}. {key} (Similarity: {unpack_sim(similarities[key])})

            ```json
            {str(orjson.dumps(casebase[key]))}
            ```
        """)

    return result


@dataclass(slots=True, frozen=True)
class build[K, V, S: Float, T: BaseModel | str](RagFunc[K, V, S, T]):
    generation_func: AnyGenerationFunc[T]
    prompt: str
    prompt_template: Callable[[str, Casebase[K, V], V, SimMap[K, S]], str] = (
        default_prompt_template
    )

    def __call__(
        self, pairs: Sequence[tuple[Casebase[K, V], V, SimMap[K, S]]]
    ) -> Sequence[T]:
        func = GenerationSeqWrapper(self.generation_func)
        prompts = [self.prompt_template(self.prompt, *pair) for pair in pairs]

        return func(prompts)
