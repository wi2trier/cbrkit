from collections.abc import Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any

from ..encoders import json_markdown
from ..genai._model import DocumentsPrompt
from ..helpers import sim_map2ranking, unpack_float
from ..typing import (
    Casebase,
    ConversionFunc,
    Float,
    JsonEntry,
    PoolingPromptFunc,
    PromptFunc,
    SimMap,
)

__all__ = [
    "transpose",
    "default",
    "documents_aware",
    "pooling",
]


@dataclass(slots=True, frozen=True)
class transpose[P, K, V1, V2, S: Float](PromptFunc[P, K, V1, S]):
    prompt_func: PromptFunc[P, K, V2, S]
    conversion_func: ConversionFunc[V1, V2]

    def __call__(
        self,
        casebase: Casebase[K, V1],
        query: V1,
        similarities: SimMap[K, S] | None,
    ) -> P:
        return self.prompt_func(
            {key: self.conversion_func(value) for key, value in casebase.items()},
            self.conversion_func(query),
            similarities,
        )


@dataclass(slots=True, frozen=True)
class default[V](PromptFunc[str, Any, V, Float]):
    instructions: str | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=json_markdown)
    metadata: JsonEntry | None = None

    def __call__(
        self,
        casebase: Casebase[Any, V],
        query: V,
        similarities: SimMap[Any, Float] | None,
    ) -> str:
        result = ""

        if self.instructions is not None:
            result += self.instructions

        result = dedent(f"""
            ## Query

            {self.encoder(query)}

            ## Documents Collection
        """)

        ranking = (
            sim_map2ranking(similarities)
            if similarities is not None
            else list(casebase.keys())
        )

        for rank, key in enumerate(ranking, start=1):
            if similarities is not None:
                result += dedent(f"""
                    ### {rank}. {key} (Similarity: {unpack_float(similarities[key]):.3f})
                """)
            else:
                result += dedent(f"""
                    ### {key}
                """)

            result += dedent(f"""
                {self.encoder(casebase[key])}
            """)

        if self.metadata is not None:
            result += dedent(f"""
                ## Metadata

                {self.encoder(self.metadata)}
            """)

        return result


@dataclass(slots=True, frozen=True)
class documents_aware[V](PromptFunc[DocumentsPrompt[str], Any, V, Any]):
    instructions: str | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=json_markdown)
    metadata: JsonEntry | None = None

    def __call__(
        self,
        casebase: Casebase[Any, V],
        query: V,
        similarities: SimMap[Any, Float] | None,
    ) -> DocumentsPrompt:
        result = ""

        if self.instructions is not None:
            result += self.instructions

        result = dedent(f"""
            ## Query

            {self.encoder(query)}
        """)

        if self.metadata is not None:
            result += dedent(f"""
                ## Metadata

                {self.encoder(self.metadata)}
            """)

        ranking = (
            sim_map2ranking(similarities)
            if similarities is not None
            else list(casebase.keys())
        )

        return DocumentsPrompt(
            result,
            {
                key: {
                    "text": self.encoder(casebase[key]),
                    "similarity": f"{unpack_float(similarities[key]):.3f}",
                    "rank": str(rank),
                }
                if similarities is not None
                else {
                    "text": self.encoder(casebase[key]),
                }
                for rank, key in enumerate(ranking)
            },
        )


@dataclass(slots=True, frozen=True)
class pooling[V](PoolingPromptFunc[str, V]):
    instructions: str | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=json_markdown)
    metadata: JsonEntry | None = None

    def __call__(
        self,
        values: Sequence[V],
    ) -> str:
        result = ""

        if self.instructions is not None:
            result += self.instructions

        result = dedent("""
            ## Documents Collection
        """)

        for value in values:
            result += dedent(f"""
                {self.encoder(value)}
            """)

        if self.metadata is not None:
            result += dedent(f"""
                ## Metadata

                {self.encoder(self.metadata)}
            """)

        return result
