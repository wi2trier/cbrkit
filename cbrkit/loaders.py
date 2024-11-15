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

from .helpers import load_object
from .typing import Casebase, FilePath

__all__ = [
    "csv",
    "polars",
    "file",
    "folder",
    "json",
    "path",
    "toml",
    "yaml",
    "python",
    "txt",
    "xml",
]

python = load_object

try:
    import pandas as pd

    @dataclass(slots=True, frozen=True)
    class pandas(Mapping[int, pd.Series]):
        df: pd.DataFrame

        def __getitem__(self, key: int | str) -> pd.Series:
            if isinstance(key, str):
                return cast(pd.Series, self.df.loc[key])
            elif isinstance(key, int):
                return self.df.iloc[key]

            raise TypeError(f"Invalid key type: {type(key)}")

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
    """Converts a path into a Casebase. The path can be a folder or a file.

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

    cb: Casebase[Any, Any] | None = None

    if path.is_file():
        cb = file(path)
    elif path.is_dir():
        cb = folder(path, pattern or "**/*")
    else:
        raise FileNotFoundError(path)

    if cb is None:
        raise NotImplementedError()

    return cb


def file(path: Path) -> Casebase[Any, Any] | None:
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
    if path.suffix not in _batch_loaders:
        return None

    loader = _batch_loaders[path.suffix]
    cb = loader(path)

    return cb


def folder(path: Path, pattern: str) -> Casebase[Any, Any] | None:
    """Converts the files of a folder into a Casebase. The files can be of type txt, csv, json, toml, yaml, or yml.

    Args:
        path: Path of the folder.
        pattern: Relative pattern for the files.

    Returns:
        Returns a Casebase.

    Examples:
        >>> from pathlib import Path
        >>> folder_path = Path("./data")
        >>> result = folder(folder_path, "*.csv")
        >>> assert result is not None
    """
    cb: Casebase[Any, Any] = {}

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _single_loaders:
            loader = _single_loaders[file.suffix]
            cb[file.name] = loader(file)

    if len(cb) == 0:
        return None

    return cb


try:
    from pydantic import BaseModel

    def validate(data: Casebase[Any, Any], validation_model: BaseModel):
        """Validates the data against a Pydantic model.

        Throws a ValueError if data is None or a Pydantic ValidationError if the data does not match the model.

        Args:
            data: Data to validate. Can be an entire case base or a single case.
            validation_model: Pydantic model to validate the data.

        Examples:
            >>> from pydantic import BaseModel, PositiveInt, NonNegativeInt
            >>> from data.cars_validation_model import Car
            >>> from pathlib import Path
            >>> data = path(Path("data/cars-1k.csv"))
            >>> validate(data, Car)
            >>> import polars as pl
            >>> df = pl.read_csv("data/cars-1k.csv")
            >>> data = polars(df)
            >>> validate(data, Car)
        """

        for item in data.values():
            validate_single(item, validation_model)

    def validate_single(data: Any, validation_model: BaseModel):
        """Validates a single case against a Pydantic model.

        Throws a ValueError if data is None or a Pydantic ValidationError if the data does not match the model.

        Args:
            data: Data to validate. Can be a single case.
            validation_model: Pydantic model to validate the data.
        """
        if data is None:
            raise ValueError("Data is None")

        validation_model.model_validate(data)

    __all__ += ["validate", "validate_single"]

except ImportError:
    pass
