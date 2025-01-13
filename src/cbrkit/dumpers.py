from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import rtoml
import polars as pl
import yaml as yamllib

from .typing import ConversionFunc, FilePath

__all__ = [
    "json_markdown",
    "json",
    "file",
    "directory",
    "path",
    "toml",
    "csv",
    "yaml",
]


@dataclass(slots=True, frozen=True)
class toml(ConversionFunc[Any, str]):
    """Writes an object to toml."""

    def __call__(self, obj: Any) -> str:
        obj = {f"i{k}": v for k, v in obj.items()}
        return rtoml.dumps(obj)


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
    def __flatten_dict(nested_dict: dict) -> list[dict[str, Any]]:
        flattened = []

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

    def __call__(self, obj: Any) -> str:
        return yamllib.dump(obj)


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
class json(ConversionFunc[Any, bytes]):
    """Writes an object to json bytes.

    Args:
        default: Function to serialize arbitrary objects, see orjson documentation.
        option: Serialization options, see orjson documentation.
            Multiple options can be combined using the bitwise OR operator `|`.
    """

    default: Callable[[Any], Any] | None = None
    option: int | None = None

    def __call__(self, obj: Any) -> bytes:
        if isinstance(obj, dict) and any(
            not isinstance(obj, str) for t in (dict, list)
        ):
            obj = {str(k): v for k, v in obj.items()}
        return orjson.dumps(obj, default=self.default, option=self.option)


Dumper = Callable[[Any], str | bytes]

dumpers: dict[str, Dumper] = {
    ".json": json(),
    ".toml": toml(),
    ".csv": csv(),
    ".yaml": yaml(),
}


def file(
    data: Any,
    path: FilePath,
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
    data: Mapping[str, Any],
    path: FilePath,
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
        file(value, path / key)


def path(
    data: Any,
    path: FilePath,
) -> None:
    """Writes arbitrary data to a file or directory.

    If the data is a mapping, it will be written to a directory.
    Otherwise, it will be written to a file.

    Args:
        data: Data to write to the file or directory.
        path: Path of the output file or directory.
    """

    if isinstance(data, Mapping):
        directory(data, path)
    else:
        file(data, path)
