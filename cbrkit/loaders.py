import csv as csvlib
import tomllib
from collections import abc
from collections.abc import Callable, Iterator
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import orjson
import pandas as pd
import xmltodict
import yaml as yamllib
from pandas import DataFrame, Series

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


class DataFrameCasebase(abc.Mapping):
    __slots__ = ("df",)

    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def __getitem__(self, key: str | int) -> Series:
        if isinstance(key, int):
            return self.df.iloc[key]
        elif isinstance(key, str):
            return self.df.loc[key]

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
        >>> df = pd.read_csv(casebase_file)
        >>> casebase = cbrkit.loaders.dataframe(df)
    """
    return DataFrameCasebase(df)


def csv(path: FilePath) -> dict[int, dict[str, str]]:
    """Reads a csv file and converts it into a dict representation

    Args:
        path: File path of the csv file

    Returns:
        Dict representation of the csv file.

    Examples:
        >>> filePath = "data/cars-1k.csv"
        >>> cbrkit.loaders.csv(filePath)
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
        >>> filePath = "data/cars-1k.json"
        >>> cbrkit.loaders.json(filePath)
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
        >>> filePath = "data/file.toml"
        >>> cbrkit.loaders.toml(filePath)
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
        >>> filePath = "data/file.yaml"
        >>> cbrkit.loaders.yaml(filePath)
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
        >>> filePath = "data/file.txt"
        >>> cbrkit.loaders.txt(filePath)
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
        >>> filePath = "data/file.xml"
        >>> cbrkit.loaders.xml(filePath)
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
        >>> jsonFile = "data/file.json"
        >>> yamlFile = "data/file.yaml"
        >>> cbrkit.loaders.toml(jsonFile)
        >>> cbrkit.loaders.toml(yamlFile)
    """
    if isinstance(path, str):
        path = Path(path)

    if path.suffix not in _data_loaders:
        raise NotImplementedError()

    loader = _data_loaders[path.suffix]
    return loader(path)


def path(path: FilePath, pattern: str | None = None) -> Casebase[Any, Any]:
    """Converts a file into a Casebase. The file can be of type

    Args:
        path: Path of the file.

    Returns:
        Returns a Casebase.

    Examples:
        >>> casebase_file = "file.csv"
        >>> df = pd.read_csv(casebase_file)
        >>> casebase = cbrkit.loaders.dataframe(df)
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
        >>> p = Path("file.csv")
        >>> cbrkit.loaders.file(p)
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
        >>> p = Path("foldername")
        >>> cbrkit.loaders.folder(p)
    """
    cb: Casebase[Any, Any] = {}

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _single_loaders:
            loader = _single_loaders[path.suffix]
            cb[file.name] = loader(file)

    if len(cb) == 0:
        return None

    return cb
