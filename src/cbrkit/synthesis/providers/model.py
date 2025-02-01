import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ...helpers import event_loop, get_logger
from ...typing import BatchConversionFunc, StructuredValue

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(slots=True, frozen=True)
class ChatPrompt[P](StructuredValue[P]):
    messages: Sequence[ChatMessage]


@dataclass(slots=True, frozen=True)
class DocumentsPrompt[P](StructuredValue[P]):
    documents: Mapping[str, Mapping[str, str]]


@dataclass(slots=True, kw_only=True)
class BaseProvider[P, R](BatchConversionFunc[P, R], ABC):
    model: str
    response_type: type[R]
    delay: float = 0
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, batches: Sequence[P]) -> Sequence[R]:
        return event_loop.get().run_until_complete(self.__call_batches__(batches))

    async def __call_batches__(self, batches: Sequence[P]) -> Sequence[R]:
        logger.info(f"Processing {len(batches)} batches with {self.model}")

        return await asyncio.gather(
            *(
                self.__call_batch_with_delay__(batch, idx)
                for idx, batch in enumerate(batches)
            )
        )

    async def __call_batch_with_delay__(self, prompt: P, idx: int) -> R:
        if self.delay > 0:
            await asyncio.sleep(idx * self.delay)

        result = await self.__call_batch__(prompt)
        logger.debug(f"Result of batch {idx + 1}: {result}")

        return result

    @abstractmethod
    async def __call_batch__(self, prompt: P) -> R: ...


@dataclass(slots=True, kw_only=True)
class ChatProvider[P, R](BaseProvider[P, R], ABC):
    system_message: str | None = None
    messages: Sequence[ChatMessage] = field(default_factory=tuple)
