from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..dumpers import markdown
from ..helpers import get_value, sim_map2ranking, unpack_float, unpack_value
from ..typing import (
    Casebase,
    ConversionFunc,
    Float,
    JsonEntry,
    SimMap,
    StructuredValue,
    SynthesizerPromptFunc,
)
from .providers.model import DocumentsPrompt

__all__ = [
    "concat",
    "transpose",
    "transpose_value",
    "default",
    "documents_aware",
    "pooling",
]


@dataclass(slots=True, frozen=True)
class concat[K, V, S: Float](SynthesizerPromptFunc[str, K, V, S]):
    prompts: Sequence[SynthesizerPromptFunc[str, K, V, S] | str]
    separator: str = "\n\n"

    def __call__(
        self,
        casebase: Casebase[K, V],
        query: V | None,
        similarities: SimMap[K, S] | None,
    ) -> str:
        return self.separator.join(
            prompt if isinstance(prompt, str) else prompt(casebase, query, similarities)
            for prompt in self.prompts
        )


@dataclass(slots=True, frozen=True)
class transpose[P, K, V1, V2, S: Float](SynthesizerPromptFunc[P, K, V1, S]):
    prompt_func: SynthesizerPromptFunc[P, K, V2, S]
    conversion_func: ConversionFunc[V1, V2]

    def __call__(
        self,
        casebase: Casebase[K, V1],
        query: V1 | None,
        similarities: SimMap[K, S] | None,
    ) -> P:
        return self.prompt_func(
            {key: self.conversion_func(value) for key, value in casebase.items()},
            self.conversion_func(query) if query is not None else None,
            similarities,
        )


def transpose_value[P, K, V, S: Float](
    func: SynthesizerPromptFunc[P, K, V, S],
) -> SynthesizerPromptFunc[P, K, StructuredValue[V], S]:
    return transpose(func, get_value)


def encode[T](value: T, encoder: ConversionFunc[T, str]) -> str:
    if value is None:
        return ""
    elif isinstance(value, str):
        return value
    elif isinstance(value, int | float | bool):
        return str(value)

    return encoder(value)


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

    instructions: str | SynthesizerPromptFunc[str, Any, V, Float] | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=markdown)
    metadata: JsonEntry | None = None

    def __call__(
        self,
        casebase: Casebase[Any, V],
        query: V | None,
        similarities: SimMap[Any, Float] | None,
    ) -> str:
        result = ""

        if isinstance(self.instructions, Callable):
            result += self.instructions(casebase, query, similarities)
        elif isinstance(self.instructions, str):
            result += self.instructions

        if query is not None:
            result += f"""
## Query

{encode(query, self.encoder)}
"""

        result += """
## Documents Collection
"""

        ranking = (
            sim_map2ranking(similarities)
            if similarities is not None
            else list(casebase.keys())
        )

        for rank, key in enumerate(ranking, start=1):
            if similarities is not None:
                result += f"""
### {key} (Rank: {rank}, Similarity: {unpack_float(similarities[key]):.3f})
"""
            else:
                result += f"""
### {key}
"""

            result += f"""
{encode(casebase[key], self.encoder)}
"""

        if self.metadata is not None:
            result += f"""
## Metadata

{encode(self.metadata, self.encoder)}
"""

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

    instructions: str | SynthesizerPromptFunc[str, Any, V, Float] | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=markdown)
    metadata: JsonEntry | None = None

    def __call__(
        self,
        casebase: Casebase[Any, V],
        query: V | None,
        similarities: SimMap[Any, Float] | None,
    ) -> DocumentsPrompt:
        result = ""

        if isinstance(self.instructions, Callable):
            result += self.instructions(casebase, query, similarities)
        elif isinstance(self.instructions, str):
            result += self.instructions

        if query is not None:
            result += f"""
## Query

{encode(query, self.encoder)}
"""

        if self.metadata is not None:
            result += f"""
## Metadata

{encode(self.metadata, self.encoder)}
"""

        ranking = (
            sim_map2ranking(similarities)
            if similarities is not None
            else list(casebase.keys())
        )

        return DocumentsPrompt(
            result,
            {
                key: {
                    "text": encode(casebase[key], self.encoder),
                    "similarity": f"{unpack_float(similarities[key]):.3f}",
                    "rank": str(rank),
                }
                if similarities is not None
                else {
                    "text": encode(casebase[key], self.encoder),
                }
                for rank, key in enumerate(ranking)
            },
        )


@dataclass(slots=True, frozen=True)
class pooling[V](ConversionFunc[Sequence[V], str]):
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

    instructions: str | ConversionFunc[Sequence[V], str] | None = None
    encoder: ConversionFunc[V | JsonEntry, str] = field(default_factory=markdown)
    metadata: JsonEntry | None = None
    unpack: bool = True

    def __call__(
        self,
        values: Sequence[V],
    ) -> str:
        result = ""

        if isinstance(self.instructions, Callable):
            result += self.instructions(values)
        elif isinstance(self.instructions, str):
            result += self.instructions

        result += """
## Partial Results
"""

        for idx, value in enumerate(values, start=1):
            if self.unpack:
                value = unpack_value(value)

            result += f"""
### Result {idx}

{encode(value, self.encoder)}
"""

        if self.metadata is not None:
            result += f"""
## Metadata

{encode(self.metadata, self.encoder)}
"""

        return result
