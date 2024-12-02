from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

from attr import dataclass

from ..typing import StructuredValue


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class ChatPrompt[P](StructuredValue[P]):
    value: P
    messages: Sequence[ChatMessage]


@dataclass(slots=True, frozen=True)
class DocumentsPrompt[P](StructuredValue[P]):
    value: P
    documents: Mapping[str, str | Mapping[str, str]]
