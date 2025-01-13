from collections.abc import Sequence
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any

from ..dumpers import json_markdown
from ..helpers import sim_map2ranking, unpack_float
from ..typing import (
    Casebase,
    ConversionFunc,
    ConversionPoolingFunc,
    Float,
    JsonEntry,
    SimMap,
    SynthesizerPromptFunc,
)
from .providers.model import DocumentsPrompt

__all__ = [
    "transpose",
    "default",
    "documents_aware",
    "pooling",
]


@dataclass(slots=True, frozen=True)
class transpose[P, K, V1, V2, S: Float](SynthesizerPromptFunc[P, K, V1, S]):
    prompt_func: SynthesizerPromptFunc[P, K, V2, S]
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
class default[V](SynthesizerPromptFunc[str, Any, V, Float]):
    """Produces an LLM input which provides context for the LLM to be able to perform instructions.

    Args:
        instructions: Instructions for the LLM to execute on the input.
        encoder: Encoder function to convert the a case or query to a string.
        metadata: Optional metadata to include in the prompt.

    Returns:
        A string to be used as an LLM input.

    Examples:
        >>> prompt = default("Give me a summary of the found cars.")
        >>> prompt(casebase, query, similarities) # doctest: +SKIP
    """

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

        result += dedent(f"""
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
                    ### {key} (Rank: {rank}, Similarity: {unpack_float(similarities[key]):.3f})
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
class documents_aware[V](SynthesizerPromptFunc[DocumentsPrompt[str], Any, V, Any]):
    """
    Produces a structured LLM input (as of now: exclusive for cohere) which provides context for the LLM to be able to perform instructions.

    Args:
        instructions: Instructions for the LLM to execute on the input.
        encoder: Encoder function to convert the a case or query to a string.
        metadata: Optional metadata to include in the prompt.

    Examples:
        >>> prompt = documents_aware("Give me a summary of the found cars.")
        >>> prompt(casebase, query, similarities) # doctest: +SKIP
    """

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
class pooling[V](ConversionPoolingFunc[V, str]):
    """
    Produces an LLM input to aggregate partial results (i.e., the LLM output for single chunks) to a final, global result.

    Args:
        instructions: Instructions for the LLM to execute on the input.
        encoder: Encoder function to convert the a case or query to a string.
        metadata: Optional metadata to include in the prompt.

    Examples:
        >>> prompt = pooling("Please find the best match from the following partial results.")
        >>> prompt([partial1, partial2, partial3]) # doctest: +SKIP
    """

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
