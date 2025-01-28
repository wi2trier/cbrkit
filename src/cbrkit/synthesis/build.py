from collections.abc import Sequence
from dataclasses import dataclass

from ..helpers import batchify_conversion, chunkify, unbatchify_conversion
from ..typing import (
    AnyConversionFunc,
    BatchPoolingFunc,
    Casebase,
    ConversionFunc,
    ConversionPoolingFunc,
    Float,
    MaybeFactory,
    SimMap,
    SynthesizerFunc,
    SynthesizerPromptFunc,
)


@dataclass(slots=True, frozen=True)
class chunks[R, K, V, S: Float](SynthesizerFunc[R, K, V, S]):
    synthesis_func: SynthesizerFunc[R, K, V, S]
    pooling_func: BatchPoolingFunc[R]
    chunk_size: int

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]]
    ) -> Sequence[R]:
        chunked_batches: list[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]] = []
        batch2chunk_indexes: list[list[int]] = []

        for casebase, query, similarities in batches:
            key_chunks = list(chunkify(list(casebase.keys()), self.chunk_size))
            last_idx = len(batch2chunk_indexes)
            batch2chunk_indexes.append(
                list(range(last_idx, last_idx + len(key_chunks)))
            )
            chunked_batches.extend(
                [
                    (
                        {key: casebase[key] for key in chunk},
                        query,
                        {key: similarities[key] for key in chunk}
                        if similarities is not None
                        else None,
                    )
                    for chunk in key_chunks
                ]
            )

        results = self.synthesis_func(chunked_batches)

        # now reconstruct the original batches to apply the pooling function
        results_per_batch: list[Sequence[R]] = [
            [results[idx] for idx in chunk_indexes]
            for chunk_indexes in batch2chunk_indexes
        ]

        return self.pooling_func(results_per_batch)


@dataclass(slots=True, frozen=True)
class pooling[P, R](BatchPoolingFunc[R]):
    generation_func: AnyConversionFunc[P, R]
    prompt_func: ConversionPoolingFunc[R, P]

    def __call__(self, batches: Sequence[Sequence[R]]) -> Sequence[R]:
        func = batchify_conversion(self.generation_func)
        prompts = [self.prompt_func(batch) for batch in batches]

        return func(prompts)


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

    def __call__(
        self, batches: Sequence[tuple[Casebase[K, V], V | None, SimMap[K, S] | None]]
    ) -> Sequence[R]:
        func = batchify_conversion(self.generation_func)

        return func([self.prompt_func(*batch) for batch in batches])
