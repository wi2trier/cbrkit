from collections.abc import Callable, Mapping
from typing import Any

import orjson

from .typing import FilePath

__all__ = ["json"]


def json(
    data: Mapping[Any, Any],
    path: FilePath,
    default: Callable[[Any], Any] | None = None,
    option: int | None = None,
) -> None:
    """Writes a dict to a json file.

    Args:
        data: Dict to write to the file.
        path: Path of the output file.
        default: Function to serialize arbitrary objects, see orjson documentation.
        option: Serialization options, see orjson documentation.
            Multiple options can be combined using the bitwise OR operator `|`.

    """

    # This is a workaround to force the dict conversion of wrapper classes (e.g., for polars)
    data_dict = {key: value for key, value in data.items()}

    with open(path, "wb") as f:
        f.write(orjson.dumps(data_dict, default=default, option=option))
