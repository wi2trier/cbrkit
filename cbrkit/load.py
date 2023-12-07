import csv
import tomllib
from collections import abc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
import yaml
from pandas import DataFrame, Series

from cbrkit import model

__all__ = ("load_path", "load_dataframe", "import_string")


def import_string(import_name: str, silent: bool = False) -> Any:
    """Imports an object based on a string.  This is useful if you want to
    use import paths as endpoints or something similar.  An import path can
    be specified either in dotted notation (``xml.sax.saxutils.escape``)
    or with a colon as object delimiter (``xml.sax.saxutils:escape``).

    If the `silent` is True the return value will be `None` if the import
    fails.

    :return: imported object
    """
    try:
        if ":" in import_name:
            module, obj = import_name.split(":", 1)
        elif "." in import_name:
            module, _, obj = import_name.rpartition(".")
        else:
            return __import__(import_name)
        return getattr(__import__(module, None, None, [obj]), obj)
    except (ImportError, AttributeError):
        if not silent:
            raise


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


def load_dataframe(df: DataFrame) -> model.Casebase[pd.Series]:
    return DataFrameCasebase(df)


def _load_csv(path: model.FilePath) -> dict[int, dict[str, str]]:
    data: dict[int, dict[str, str]] = {}

    with open(path) as fp:
        reader = csv.DictReader(fp)
        row: dict[str, str]

        for idx, row in enumerate(reader):
            data[idx] = row

        return data


def _load_json(path: model.FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return orjson.loads(fp.read())


def _load_toml(path: model.FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def _load_yaml(path: model.FilePath) -> dict[str, Any]:
    data: dict[str, Any] = {}

    with open(path, "rb") as fp:
        for doc in yaml.safe_load_all(fp):
            data |= doc

    return data


def _load_txt(path: model.FilePath) -> str:
    with open(path) as fp:
        return fp.read()


DataLoader = Callable[[model.FilePath], dict[str, Any]]
SingleLoader = Callable[[model.FilePath], Any]
BatchLoader = Callable[[model.FilePath], dict[Any, Any]]

data_loaders: dict[str, DataLoader] = {
    ".json": _load_json,
    ".toml": _load_toml,
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
}

# They contain the whole casebase in one file
_batch_loaders: dict[str, BatchLoader] = {
    **data_loaders,
    ".csv": _load_csv,
}

# They contain one case per file
# Since structured formats may also be used for single cases, they are also included here
_single_loaders: dict[str, SingleLoader] = {
    **_batch_loaders,
    ".txt": _load_txt,
}


def load_path(path: model.FilePath, pattern: str | None = None) -> model.Casebase[Any]:
    if isinstance(path, str):
        path = Path(path)

    cb: model.Casebase[Any] | None = None

    if path.is_file():
        cb = load_file(path)
    elif path.is_dir():
        cb = load_folder(path, pattern or "**/*")
    else:
        raise FileNotFoundError(path)

    if cb is None:
        raise NotImplementedError()

    return cb


def load_file(path: Path) -> model.Casebase[Any] | None:
    if path.suffix not in _batch_loaders:
        return None

    loader = _batch_loaders[path.suffix]
    cb = loader(path)

    return cb


def load_folder(path: Path, pattern: str) -> model.Casebase[Any] | None:
    cb: model.Casebase[Any] = {}

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _single_loaders:
            loader = _single_loaders[path.suffix]
            cb[file.name] = loader(file)

    if len(cb) == 0:
        return None

    return cb
