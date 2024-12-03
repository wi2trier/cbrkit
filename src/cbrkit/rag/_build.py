from collections.abc import Sequence
from dataclasses import dataclass

from ..helpers import batchify_generation, chunkify
from ..typing import (
    AnyGenerationFunc,
    BatchPoolingFunc,
    Casebase,
    ConversionFunc,
    Float,
    PoolingPromptFunc,
    PromptFunc,
    RagFunc,
    SimMap,
)


@dataclass(slots=True, frozen=True)
class chunks[R, K, V, S: Float](RagFunc[R, K, V, S]):
    rag_func: RagFunc[R, K, V, S]
    pooling_func: BatchPoolingFunc[R]
    chunk_size: int

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]]
    ) -> Sequence[R]:
        result_chunks: list[Sequence[R]] = []

        for casebase, query, similarities in batches:
            key_chunks = chunkify(list(casebase.keys()), self.chunk_size)
            rag_batches = [
                (
                    {key: casebase[key] for key in chunk},
                    query,
                    {key: similarities[key] for key in chunk}
                    if similarities is not None
                    else None,
                )
                for chunk in key_chunks
            ]

            result_chunks.append(self.rag_func(rag_batches))

        return self.pooling_func(result_chunks)


@dataclass(slots=True, frozen=True)
class pooling[P, R](BatchPoolingFunc[R]):
    prompt_func: PoolingPromptFunc[P, R]
    generation_func: AnyGenerationFunc[P, R]

    def __call__(self, batches: Sequence[Sequence[R]]) -> Sequence[R]:
        func = batchify_generation(self.generation_func)
        prompts = [self.prompt_func(batch) for batch in batches]

        return func(prompts)


@dataclass(slots=True, frozen=True)
class transpose[R1, R2, K, V, S: Float](RagFunc[R1, K, V, S]):
    rag_func: RagFunc[R2, K, V, S]
    conversion_func: ConversionFunc[R2, R1]

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]]
    ) -> Sequence[R1]:
        return [self.conversion_func(batch) for batch in self.rag_func(batches)]


@dataclass(slots=True, frozen=True)
class build[P, R, K, V, S: Float](RagFunc[R, K, V, S]):
    generation_func: AnyGenerationFunc[P, R]
    prompt_func: PromptFunc[P, K, V, S]

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V, SimMap[K, S] | None]]
    ) -> Sequence[R]:
        func = batchify_generation(self.generation_func)

        return func([self.prompt_func(*batch) for batch in batches])
