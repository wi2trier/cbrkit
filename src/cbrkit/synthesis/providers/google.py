from dataclasses import InitVar, dataclass, field
from typing import cast, override

from pydantic import BaseModel

from ...helpers import optional_dependencies
from .model import BaseProvider, Response

with optional_dependencies():
    from google.genai import Client
    from google.genai.types import GenerateContentConfig

    type GooglePrompt = str

    @dataclass(slots=True)
    class google[R: BaseModel | str](BaseProvider[GooglePrompt, R]):
        system_message: str | None = None
        client: Client = field(default_factory=Client, repr=False)
        config: GenerateContentConfig = field(init=False)
        base_config: InitVar[GenerateContentConfig | None] = None

        def __post_init__(self, base_config: GenerateContentConfig | None) -> None:
            self.config = base_config or GenerateContentConfig()

            if issubclass(self.response_type, BaseModel):
                self.config.response_schema = self.response_type

            if self.system_message is not None:
                self.config.system_instruction = self.system_message

        @override
        async def __call_batch__(self, prompt: GooglePrompt) -> Response[R]:
            res = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.config,
                **self.extra_kwargs,
            )

            if (
                issubclass(self.response_type, BaseModel)
                and (parsed := res.parsed)
                and isinstance(parsed, self.response_type)
            ):
                return Response(cast(R, parsed))

            elif issubclass(self.response_type, str) and (text := res.text):
                return Response(cast(R, text))

            raise ValueError("Invalid response", res)
