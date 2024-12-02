from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import orjson

from .typing import ConversionFunc

__all__ = ["json_markdown", "json_bytes"]


@dataclass(slots=True, frozen=True)
class json_markdown(ConversionFunc[Any, str]):
    """Writes an object to a json code block in markdown.

    Args:
        default: Function to serialize arbitrary objects, see orjson documentation.
        option: Serialization options, see orjson documentation.
            Multiple options can be combined using the bitwise OR operator `|`.
    """

    default: Callable[[Any], Any] | None = None
    option: int | None = None

    def __call__(self, obj: Any) -> str:
        return f"""```json
{str(orjson.dumps(obj, default=self.default, option=self.option))}
```"""


@dataclass(slots=True, frozen=True)
class json_bytes(ConversionFunc[Any, bytes]):
    """Writes an object to json bytes.

    Args:
        default: Function to serialize arbitrary objects, see orjson documentation.
        option: Serialization options, see orjson documentation.
            Multiple options can be combined using the bitwise OR operator `|`.
    """

    default: Callable[[Any], Any] | None = None
    option: int | None = None

    def __call__(self, obj: Any) -> bytes:
        return orjson.dumps(obj, default=self.default, option=self.option)
