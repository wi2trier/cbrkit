"""
This module provides several loaders to read data from different file formats and convert it into a Casebase. To validate the data against a Pydantic model, a `validate` function is also provided.
"""

import csv as csvlib
import tomllib
from collections.abc import Callable, Iterator, Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import orjson
import pandas as pd
import xmltodict
import yaml as yamllib
from pandas import DataFrame, Series
from pydantic import BaseModel

from cbrkit.typing import Casebase, FilePath

__all__ = [
    "csv",
    "dataframe",
    "file",
    "folder",
    "json",
    "path",
    "toml",
    "yaml",
    "python",
    "txt",
    "xml",
    "validate",
]


def python(import_name: str) -> Any:
    """Import an object based on a string.

    Args:
        import_name: Can either be in in dotted notation (`module.submodule.object`)
            or with a colon as object delimiter (`module.submodule:object`).

    Returns:
        The imported object.
    """

    if ":" in import_name:
        module_name, obj_name = import_name.split(":", 1)
    elif "." in import_name:
        module_name, obj_name = import_name.rsplit(".", 1)
    else:
        raise ValueError(f"Failed to import {import_name!r}")

    module = import_module(module_name)

    return getattr(module, obj_name)


class DataFrameCasebase(Mapping):
    __slots__ = ("df",)

    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def __getitem__(self, key: str | int) -> Series:
        if isinstance(key, int):
            return self.df.iloc[key]
        elif isinstance(key, str):
            return cast(Series, self.df.loc[key])

        raise TypeError(f"Invalid key type: {type(key)}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.df.index)

    def __len__(self) -> int:
        return len(self.df)


def dataframe(df: DataFrame) -> Casebase[Any, pd.Series]:
    """Converts a pandas DataFrame into a Casebase.

    Args:
        df: pandas DataFrame.

    Returns:
        Returns a Casebase as a DataFrameCasebase.

    Examples:
        >>> file_path = "./data/cars-1k.csv"
        >>> df = pd.read_csv(file_path)
        >>> result = dataframe(df)
    """
    return DataFrameCasebase(df)


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


def _csv_pandas(path: FilePath) -> dict[int, pd.Series]:
    df = pd.read_csv(path)

    return cast(dict[int, pd.Series], dataframe(df))


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


DataLoader = Callable[[FilePath], dict[str, Any]]
SingleLoader = Callable[[FilePath], Any]
BatchLoader = Callable[[FilePath], dict[Any, Any]]

_data_loaders: dict[str, DataLoader] = {
    ".json": json,
    ".toml": toml,
    ".yaml": yaml,
    ".yml": yaml,
}

# They contain the whole casebase in one file
_batch_loaders: dict[str, BatchLoader] = {
    **_data_loaders,
    ".csv": _csv_pandas,
}

# They contain one case per file
# Since structured formats may also be used for single cases, they are also included here
_single_loaders: dict[str, SingleLoader] = {
    **_batch_loaders,
    ".txt": txt,
}


def data(path: FilePath) -> dict[str, Any]:
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


def validate(data: Casebase[Any, Any] | Any, validation_model: BaseModel):
    """Validates the data against a Pydantic model. Throws a ValueError if data is None or a Pydantic ValidationError if the data does not match the model.

    Args:
        data: Data to validate. Can be an entire case base or a single case.
        validation_model: Pydantic model to validate the data.

    Examples:
        >>> from pydantic import BaseModel, PositiveInt, NonNegativeInt
        >>> from data.cars_validation_model import Car
        >>> from pathlib import Path
        >>> data = path(Path("data/cars-1k.csv"))
        >>> validate(data, Car)
        >>> import pandas as pd
        >>> df = pd.read_csv("data/cars-1k.csv")
        >>> data = dataframe(df)
        >>> validate(data, Car)
    """
    assert data is not None

    if isinstance(data, DataFrameCasebase):
        data = data.df.to_dict("index")

    if isinstance(data, Mapping):
        for item in data.values():
            validation_model.model_validate(item)
    else:
        validation_model.model_validate(data)
