from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, override

from ..helpers import (
    batchify_conversion,
    chain_map_chunks,
    chunkify_overlap,
)
from ..typing import (
    AnyConversionFunc,
    BatchConversionFunc,
    Casebase,
    ConversionFunc,
    Float,
    MaybeFactory,
    SimMap,
    SynthesizerFunc,
    SynthesizerPromptFunc,
    Value,
)


@dataclass(slots=True, frozen=True)
class chunks[R1, R2, K, V, S: Float](SynthesizerFunc[R1, K, V, S]):
    synthesis_func: SynthesizerFunc[R2, K, V, S]
    conversion_func: AnyConversionFunc[Sequence[R2], R1]
    size: int
    overlap: int = 0
    direction: Literal["left", "right", "both"] = "both"

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]]
    ) -> Sequence[R1]:
        conversion_func = batchify_conversion(self.conversion_func)

        chunked_batches = [
            [
                (
                    {key: casebase[key] for key in chunked_keys},
                    query,
                    {key: similarities[key] for key in chunked_keys}
                    if similarities is not None
                    else None,
                )
                for chunked_keys in chunkify_overlap(
                    list(casebase.keys()),
                    self.size,
                    self.overlap,
                    self.direction,
                )
            ]
            for casebase, query, similarities in batches
        ]

        results_per_batch = chain_map_chunks(chunked_batches, self.synthesis_func)

        return conversion_func(results_per_batch)


@dataclass(slots=True, frozen=True)
class pooling[P, R1, R2](BatchConversionFunc[Sequence[Value[R2]], R1]):
    generation_func: AnyConversionFunc[P, R1]
    prompt_func: AnyConversionFunc[Sequence[Value[R2]], P]

    def __call__(self, batches: Sequence[Sequence[Value[R2]]]) -> Sequence[R1]:
        generation_func = batchify_conversion(self.generation_func)
        prompt_func = batchify_conversion(self.prompt_func)
        prompts = prompt_func(batches)

        return generation_func(prompts)


@dataclass(slots=True, frozen=True)
class transpose[R1, R2, K, V, S: Float](SynthesizerFunc[R1, K, V, S]):
    synthesis_func: SynthesizerFunc[R2, K, V, S]
    conversion_func: ConversionFunc[R2, R1]

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]]
    ) -> Sequence[R1]:
        return [self.conversion_func(batch) for batch in self.synthesis_func(batches)]


@dataclass(slots=True, frozen=True)
class build[P, R, K, V, S: Float](SynthesizerFunc[R, K, V, S]):
    generation_func: MaybeFactory[AnyConversionFunc[P, R]]
    prompt_func: SynthesizerPromptFunc[P, K, V, S]

    @override
    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]]
    ) -> Sequence[R]:
        func = batchify_conversion(self.generation_func)

        return func([self.prompt_func(*batch) for batch in batches])
