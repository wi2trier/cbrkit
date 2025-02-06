from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson
import polars as pl
import rtoml
import yaml as yamllib
from pydantic import BaseModel

from .helpers import get_name
from .typing import ConversionFunc, FilePath

__all__ = [
    "markdown",
    "json",
    "file",
    "directory",
    "path",
    "toml",
    "csv",
    "yaml",
]


def default_conversion_func(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

    if hasattr(obj, "to_dict"):
        return default_conversion_func(obj.to_dict())

    if hasattr(obj, "dump"):
        return default_conversion_func(obj.dump())

    if isinstance(obj, dict) and any(not isinstance(k, str) for k in obj.keys()):
        return {str(k): v for k, v in obj.items()}

    return obj


@dataclass(slots=True, frozen=True)
class toml(ConversionFunc[Any, str]):
    """Writes an object to toml."""

    conversion_func: ConversionFunc[Any, Any] = default_conversion_func

    def __call__(self, obj: Any) -> str:
        return rtoml.dumps(self.conversion_func(obj))


@dataclass(slots=True, frozen=True)
class csv(ConversionFunc[Any, str]):
    """Writes an object to a csv file."""

    @staticmethod
    def _flatten_recursive(obj: Any, prefix: str = "") -> dict[str, Any]:
        flat_item = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                flat_item.update(csv._flatten_recursive(value, new_prefix))
        else:
            flat_item[prefix] = obj
        return flat_item

    @staticmethod
    def __flatten_dict(nested_dict: Any) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []

        # Handle both dict with numeric keys and list inputs
        items = nested_dict.values() if isinstance(nested_dict, dict) else nested_dict

        for d in items:
            flat_item = csv._flatten_recursive(d)
            flattened.append(flat_item)

        return flattened

    def __call__(self, obj: Any) -> str:
        # remove nested dicts
        if not isinstance(obj, dict):
            raise ValueError("Object must be a dictionary")
        obj = self.__flatten_dict(obj)
        df = pl.DataFrame(obj)
        return df.write_csv()


@dataclass(slots=True, frozen=True)
class yaml(ConversionFunc[Any, str]):
    """Writes an object to a csv file."""

    conversion_func: ConversionFunc[Any, Any] = default_conversion_func

    def __call__(self, obj: Any) -> str:
        return yamllib.dump(self.conversion_func(obj))


@dataclass(slots=True, frozen=True)
class json(ConversionFunc[Any, bytes]):
    """Writes an object to json bytes.

    Args:
        default: Function to serialize arbitrary objects, see orjson documentation.
        option: Serialization options, see orjson documentation.
            Multiple options can be combined using the bitwise OR operator `|`.
    """

    default: Callable[[Any], Any] | None = None
    option: int | None = None
    conversion_func: ConversionFunc[Any, Any] = default_conversion_func

    def __call__(self, obj: Any) -> bytes:
        return orjson.dumps(
            self.conversion_func(obj),
            default=self.default,
            option=self.option,
        )


Dumper = Callable[[Any], str | bytes]


dumpers: dict[str, Dumper] = {
    ".json": json(),
    ".toml": toml(),
    ".csv": csv(),
    ".yaml": yaml(),
}


@dataclass(slots=True, frozen=True)
class markdown(ConversionFunc[Any, str]):
    """Writes an object to a code block in markdown.

    Args:
        dumper: Function to serialize arbitrary objects, see orjson documentation.
        language: Language of the code block.
    """

    dumper: Dumper = field(default_factory=json)
    language: str | None = None

    def __call__(self, obj: Any) -> str:
        language = get_name(self.dumper) if self.language is None else self.language

        dumped_obj = self.dumper(obj)

        if isinstance(dumped_obj, bytes | bytearray):
            dumped_obj = dumped_obj.decode("utf-8")

        return f"```{language}\n{dumped_obj}\n```"


def file(
    path: FilePath,
    data: Any,
    dumper: Dumper | None = None,
) -> None:
    """Writes arbitrary data to a json file.

    Args:
        data: Data to write to the file.
        path: Path of the output file.
        dumper: Function to use
    """
    if isinstance(path, str):
        path = Path(path)

    if dumper is None and path.suffix not in dumpers:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if dumper is None:
        dumper = dumpers[path.suffix]

    encoded_data = dumper(data)

    if isinstance(encoded_data, str):
        with open(path, "w") as f:
            f.write(encoded_data)

    elif isinstance(encoded_data, bytes):
        with open(path, "wb") as f:
            f.write(encoded_data)

    else:
        raise ValueError("Invalid dumper output type")


def directory(
    path: FilePath,
    data: Mapping[str, Any],
):
    """Writes arbitrary data to a directory.

    Args:
        data: Data to write to the directory.
        path: Path of the output directory.
    """

    if isinstance(path, str):
        path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    for key, value in data.items():
        file(path / key, value)


def path(
    path: FilePath,
    data: Any,
) -> None:
    """Writes arbitrary data to a file or directory.

    If the data is a mapping, it will be written to a directory.
    Otherwise, it will be written to a file.

    Args:
        data: Data to write to the file or directory.
        path: Path of the output file or directory.
    """

    if isinstance(data, Mapping):
        directory(path, data)
    else:
        file(path, data)
