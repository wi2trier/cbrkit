"""
This module provides several loaders to read data from different file formats and convert it into a Casebase. To validate the data against a Pydantic model, a `validate` function is also provided.
"""

import csv as csvlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, TextIO, cast

import orjson
import pandas as pd
import polars as pl
import rtoml
import xmltodict
import yaml as yamllib
from pydantic import BaseModel

from .helpers import load_object
from .typing import Casebase, ConversionFunc, FilePath

__all__ = [
    "path",
    "file",
    "directory",
    "validate",
    "csv",
    "json",
    "polars",
    "pandas",
    "py",
    "toml",
    "txt",
    "xml",
    "yaml",
]

AnyIO = TextIO | BinaryIO
ReadableType = str | bytes | TextIO | BinaryIO


def read(data: ReadableType) -> str:
    if isinstance(data, str):
        return data

    elif isinstance(data, bytes | bytearray):
        return data.decode("utf-8")

    return read(data.read())  # pyright: ignore


@dataclass(slots=True, frozen=True)
class pandas(Mapping[int, pd.Series]):
    """A wrapper around a pandas DataFrame to provide a dict-like interface"""

    df: pd.DataFrame

    def __getitem__(self, key: int | str) -> pd.Series:
        if isinstance(key, str):
            return cast(pd.Series, self.df.loc[key])

        return cast(pd.Series, self.df.iloc[key])

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.df.shape[0]))

    def __len__(self) -> int:
        return self.df.shape[0]


@dataclass(slots=True, frozen=True)
class polars(Mapping[int, dict[str, Any]]):
    """A wrapper around a polars DataFrame to provide a dict-like interface"""

    df: pl.DataFrame

    def __getitem__(self, key: int) -> dict[str, Any]:
        return self.df.row(key, named=True)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.df.shape[0]))

    def __len__(self) -> int:
        return self.df.shape[0]


@dataclass(slots=True, frozen=True)
class py(ConversionFunc[str, Any]):
    """Reads a Python file and loads the object from it."""

    def __call__(self, source: str) -> Any:
        return load_object(source)


@dataclass(slots=True, frozen=True)
class csv(ConversionFunc[Iterable[str] | ReadableType, dict[int, dict[str, str]]]):
    """Reads a csv file and converts it into a dict representation"""

    def __call__(
        self, source: Iterable[str] | ReadableType
    ) -> dict[int, dict[str, str]]:
        if isinstance(source, ReadableType):
            source = read(source).splitlines()

        reader = csvlib.DictReader(source)  # pyright: ignore
        data: dict[int, dict[str, str]] = {}
        row: dict[str, str]

        for idx, row in enumerate(reader):
            data[idx] = row

        return data


@dataclass(slots=True, frozen=True)
class json(ConversionFunc[ReadableType, dict[Any, Any]]):
    """Reads a json file and converts it into a dict representation"""

    def __call__(self, source: ReadableType) -> dict[Any, Any]:
        data = orjson.loads(read(source))

        if isinstance(data, list):
            return dict(enumerate(data))
        elif isinstance(data, dict):
            return data

        raise TypeError(f"Invalid data type: {type(data)}")


@dataclass(slots=True, frozen=True)
class toml(ConversionFunc[ReadableType, dict[str, Any]]):
    """Reads a toml file and converts it into a dict representation"""

    def __call__(self, source: ReadableType) -> dict[str, Any]:
        return rtoml.loads(read(source))


@dataclass(slots=True, frozen=True)
class yaml(ConversionFunc[ReadableType, dict[Any, Any]]):
    """Reads a yaml file and converts it into a dict representation"""

    def __call__(self, source: ReadableType) -> dict[Any, Any]:
        data: dict[Any, Any] = {}

        for doc_idx, doc in enumerate(yamllib.safe_load_all(source)):
            if isinstance(doc, list):
                for idx, item in enumerate(doc):
                    data[doc_idx + idx] = item
            elif isinstance(doc, dict):
                data |= doc
            else:
                raise TypeError(f"Invalid document type: {type(doc)}")

        return data


@dataclass(slots=True, frozen=True)
class xml(ConversionFunc[ReadableType, dict[str, Any]]):
    """Reads a xml file and converts it into a dict representation"""

    def __call__(self, source: ReadableType) -> dict[str, Any]:
        data = xmltodict.parse(read(source))

        if len(data) == 1:
            data_without_root = data[next(iter(data))]

            return data_without_root

        return data


@dataclass(slots=True, frozen=True)
class txt(ConversionFunc[ReadableType, str]):
    """Reads a text file and converts it into a string"""

    def __call__(self, source: ReadableType) -> str:
        return read(source)


def _csv_polars(source: Path | ReadableType) -> Mapping[int, dict[str, Any]]:
    return polars(pl.read_csv(source))


StructuredLoader = Callable[[AnyIO], Mapping[Any, Any]]
AnyLoader = Callable[[AnyIO], Any]

structured_loaders: dict[str, StructuredLoader] = {
    ".json": json(),
    ".toml": toml(),
    ".yaml": yaml(),
    ".yml": yaml(),
    ".xml": xml(),
    ".csv": _csv_polars,
}

any_loaders: dict[str, AnyLoader] = {
    **structured_loaders,
    ".txt": txt(),
}


def path(
    path: FilePath, pattern: str | None = None, loader: AnyLoader | None = None
) -> Casebase[Any, Any]:
    """Converts a path into a Casebase. The path can be a directory or a file.

    Args:
        path: Path of the file.

    Returns:
        Returns a Casebase.

    Examples:
        >>> file_path = "./data/cars-1k.csv"
        >>> result = path(file_path)
    """
    if isinstance(path, str):
        path = Path(path)

    if path.is_file():
        return file(path, loader)
    elif path.is_dir():
        return directory(path, pattern)

    raise FileNotFoundError(path)


def file(path: FilePath, loader: StructuredLoader | None = None) -> Casebase[Any, Any]:
    """Converts a file into a Casebase. The file can be of type csv, json, toml, yaml, or yml.

    Args:
        path: Path of the file.

    Returns:
        Returns a Casebase.

    Examples:
        >>> from pathlib import Path
        >>> file_path = Path("./data/cars-1k.csv")
        >>> result = file(file_path)

    """
    if isinstance(path, str):
        path = Path(path)

    if loader is None and path.suffix not in structured_loaders:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    if loader is None:
        loader = structured_loaders[path.suffix]

    with path.open("rb") as fp:
        return loader(fp)


def directory(path: FilePath, pattern: str | None = None) -> Casebase[Any, Any]:
    """Converts the files of a directory into a Casebase. The files can be of type txt, csv, json, toml, yaml, or yml.

    Args:
        path: Path of the directory.
        pattern: Relative pattern for the files.

    Returns:
        Returns a Casebase.

    Examples:
        >>> from pathlib import Path
        >>> directory_path = Path("./data")
        >>> result = directory(directory_path, "*.csv")
        >>> assert result is not None
    """
    cb: Casebase[Any, Any] = {}

    if isinstance(path, str):
        path = Path(path)

    for elem in path.glob(pattern or "*"):
        if elem.is_file() and elem.suffix in any_loaders:
            loader = any_loaders[elem.suffix]

            with elem.open("rb") as fp:
                cb[elem.stem] = loader(fp)

    return cb


def validate[K, V: BaseModel](
    casebase: Casebase[K, Any], model: type[V]
) -> Casebase[K, V]:
    """Validates the casebase against a Pydantic model.

    Args:
        casebase: Casebase where the values are the data to validate.
        model: Pydantic model to validate the data.

    Examples:
        >>> from pydantic import BaseModel, NonNegativeInt
        >>> from typing import Literal
        >>> class Car(BaseModel):
        ...     price: NonNegativeInt
        ...     year: NonNegativeInt
        ...     manufacturer: str
        ...     make: str
        ...     fuel: Literal["gas", "diesel"]
        ...     miles: NonNegativeInt
        ...     title_status: Literal["clean", "rebuilt"]
        ...     transmission: Literal["automatic", "manual"]
        ...     drive: Literal["fwd", "rwd", "4wd"]
        ...     type: str
        ...     paint_color: str
        >>> data = file("data/cars-1k.csv")
        >>> casebase = validate(data, Car)
        >>> data = polars(pl.read_csv("data/cars-1k.csv"))
        >>> casebase = validate(data, Car)
    """

    return {key: model.model_validate(value) for key, value in casebase.items()}
