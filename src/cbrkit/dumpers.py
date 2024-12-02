from collections.abc import Mapping
from typing import Any

from .encoders import json_bytes
from .typing import ConversionFunc, FilePath

__all__ = ["file"]


_default_encoder = json_bytes()


def file(
    data: Mapping[Any, Any],
    path: FilePath,
    encoder: ConversionFunc[Any, bytes | str] = _default_encoder,
) -> None:
    """Writes a dict to a json file.

    Args:
        data: Dict to write to the file.
        path: Path of the output file.
        encoder: Encoder function to use
    """

    # This is a workaround to force the dict conversion of wrapper classes (e.g., for polars)
    data_dict = dict(data.items())
    encoded_data = encoder(data_dict)

    if isinstance(encoded_data, str):
        with open(path, "w") as f:
            f.write(encoded_data)
    elif isinstance(encoded_data, bytes):
        with open(path, "wb") as f:
            f.write(encoded_data)

    raise ValueError("Invalid encoder output type")
