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
    return DataFrameCasebase(df)


def csv(path: FilePath) -> dict[int, dict[str, str]]:
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


def json(path: FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return orjson.loads(fp.read())


def toml(path: FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def yaml(path: FilePath) -> dict[str, Any]:
    data: dict[str, Any] = {}

    with open(path, "rb") as fp:
        for doc in yamllib.safe_load_all(fp):
            data |= doc

    return data


def txt(path: FilePath) -> str:
    with open(path) as fp:
        return fp.read()


def xml(path: FilePath) -> dict[str, Any]:
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
    if isinstance(path, str):
        path = Path(path)

    if path.suffix not in _data_loaders:
        raise NotImplementedError()

    loader = _data_loaders[path.suffix]
    return loader(path)


def path(path: FilePath, pattern: str | None = None) -> Casebase[Any, Any]:
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
    if path.suffix not in _batch_loaders:
        return None

    loader = _batch_loaders[path.suffix]
    cb = loader(path)

    return cb


def folder(path: Path, pattern: str) -> Casebase[Any, Any] | None:
    cb: Casebase[Any, Any] = {}

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _single_loaders:
            loader = _single_loaders[path.suffix]
            cb[file.name] = loader(file)

    if len(cb) == 0:
        return None

    return cb
