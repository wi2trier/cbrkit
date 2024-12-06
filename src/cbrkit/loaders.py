"""
This module provides several loaders to read data from different file formats and convert it into a Casebase. To validate the data against a Pydantic model, a `validate` function is also provided.
"""

import csv as csvlib
import tomllib
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson
import polars as pl
import xmltodict
import yaml as yamllib
from pydantic import BaseModel

from .helpers import load_object
from .typing import Casebase, FilePath

__all__ = [
    "path",
    "file",
    "directory",
    "validate",
    "csv",
    "json",
    "polars",
    "py",
    "toml",
    "txt",
    "xml",
    "yaml",
]

py = load_object

try:
    import pandas as pd

    @dataclass(slots=True, frozen=True)
    class pandas(Mapping[int, pd.Series]):
        df: pd.DataFrame

        def __getitem__(self, key: int | str) -> pd.Series:
            if isinstance(key, str):
                return cast(pd.Series, self.df.loc[key])

            return self.df.iloc[key]

        def __iter__(self) -> Iterator[int]:
            return iter(range(self.df.shape[0]))

        def __len__(self) -> int:
            return self.df.shape[0]

    __all__ += ["pandas"]
except ImportError:
    pass


@dataclass(slots=True, frozen=True)
class polars(Mapping[int, dict[str, Any]]):
    df: pl.DataFrame

    def __getitem__(self, key: int) -> dict[str, Any]:
        return self.df.row(key, named=True)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.df.shape[0]))

    def __len__(self) -> int:
        return self.df.shape[0]


def csv(path: FilePath) -> dict[int, dict[str, str]]:
    """Reads a csv file and converts it into a dict representation

    Args:
        path: File path of the csv file

    Returns:
        Dict representation of the csv file.

    Examples:
        >>> file_path = "./data/cars-1k.csv"
        >>> result = csv(file_path)
    """
    data: dict[int, dict[str, str]] = {}

    with open(path) as fp:
        reader = csvlib.DictReader(fp)
        row: dict[str, str]

        for idx, row in enumerate(reader):
            data[idx] = row

        return data


def _csv_polars(path: FilePath) -> Mapping[int, dict[str, Any]]:
    return polars(pl.read_csv(path))


def json(path: FilePath) -> dict[Any, Any]:
    """Reads a json file and converts it into a dict representation

    Args:
        path: File path of the json file

    Returns:
        Dict representation of the json file.

    Examples:
        >>> file_path = "data/cars-1k.json"     # doctest: +SKIP
        >>> json(file_path)                     # doctest: +SKIP
    """
    with open(path, "rb") as fp:
        data = orjson.loads(fp.read())

        if isinstance(data, list):
            return dict(enumerate(data))
        elif isinstance(data, dict):
            return data
        else:
            raise TypeError(f"Invalid data type: {type(data)}")


def toml(path: FilePath) -> dict[str, Any]:
    """Reads a toml file and parses it into a dict representation

    Args:
        path: File path of the toml file

    Returns:
        Dict representation of the toml file.

    Examples:
        >>> file_path = "./data/file.toml"      # doctest: +SKIP
        >>> toml(file_path)                     # doctest: +SKIP
    """
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def yaml(path: FilePath) -> dict[Any, Any]:
    """Reads a yaml file and parses it into a dict representation

    Args:
        path: File path of the yaml file

    Returns:
        Dict representation of the yaml file.

    Examples:
        >>> file_path = "./data/cars-1k.yaml"
        >>> result = yaml(file_path)
    """
    data: dict[Any, Any] = {}

    with open(path, "rb") as fp:
        for doc_idx, doc in enumerate(yamllib.safe_load_all(fp)):
            if isinstance(doc, list):
                for idx, item in enumerate(doc):
                    data[doc_idx + idx] = item
            elif isinstance(doc, dict):
                data |= doc
            else:
                raise TypeError(f"Invalid document type: {type(doc)}")

    return data


def txt(path: FilePath) -> str:
    """Reads a text file and converts it into a string

    Args:
        path: File path of the text file

    Returns:
        String representation of the text file.

    Examples:
        >>> file_path = "data/file.txt"      # doctest: +SKIP
        >>> txt(file_path)                   # doctest: +SKIP
    """
    with open(path) as fp:
        return fp.read()


def xml(path: FilePath) -> dict[str, Any]:
    """Reads a xml file and parses it into a dict representation

    Args:
        path: File path of the xml file

    Returns:
        Dict representation of the xml file.

    Examples:
        >>> file_path = "data/file.xml"      # doctest: +SKIP
        >>> result = xml(file_path)          # doctest: +SKIP
    """
    with open(path, "rb") as fp:
        data = xmltodict.parse(fp.read())

    if len(data) == 1:
        data_without_root = data[next(iter(data))]

        return data_without_root

    return data


DataLoader = Callable[[FilePath], Mapping[str, Any]]
SingleLoader = Callable[[FilePath], Any]
BatchLoader = Callable[[FilePath], Mapping[Any, Any]]

_data_loaders: dict[str, DataLoader] = {
    ".json": json,
    ".toml": toml,
    ".yaml": yaml,
    ".yml": yaml,
}

# They contain the whole casebase in one file
_batch_loaders: dict[str, BatchLoader] = {
    **_data_loaders,
    ".csv": _csv_polars,
}

# They contain one case per file
# Since structured formats may also be used for single cases, they are also included here
_single_loaders: dict[str, SingleLoader] = {
    **_batch_loaders,
    ".txt": txt,
}


def data(path: FilePath) -> Mapping[str, Any]:
    """Reads files of types json, toml, yaml, and yml and parses it into a dict representation

    Args:
        path: Path of the file

    Returns:
        Dict representation of the file.

    Examples:
        >>> yaml_file = "./data/cars-1k.yaml"
        >>> result = data(yaml_file)
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix not in _data_loaders:
        raise NotImplementedError()

    loader = _data_loaders[path.suffix]
    return loader(path)


def path(path: FilePath, pattern: str | None = None) -> Casebase[Any, Any]:
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
        return file(path)
    elif path.is_dir():
        return directory(path, pattern or "**/*")

    raise FileNotFoundError(path)


def file(path: FilePath) -> Casebase[Any, Any]:
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

    if path.suffix not in _batch_loaders:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    loader = _batch_loaders[path.suffix]
    cb = loader(path)

    return cb


def directory(path: FilePath, pattern: str) -> Casebase[Any, Any]:
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

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _single_loaders:
            loader = _single_loaders[file.suffix]
            cb[file.name] = loader(file)

    return cb


def validate[K, V: BaseModel](casebase: Casebase[K, Any], model: V) -> Casebase[K, V]:
    """Validates the casebase against a Pydantic model.

    Args:
        casebase: Casebase where the values are the data to validate.
        model: Pydantic model to validate the data.

    Examples:
        >>> from pydantic import BaseModel, NonNegativeInt
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
        >>> validate(data, Car)
        >>> import polars as pl
        >>> df = pl.read_csv("data/cars-1k.csv")
        >>> data = polars(df)
        >>> validate(data, Car)
    """

    return {key: model.model_validate(value) for key, value in casebase.items()}
