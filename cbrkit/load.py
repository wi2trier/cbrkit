import csv
import tomllib
from collections import abc
from pathlib import Path
from typing import Any, Callable, Hashable, Iterator, cast

import orjson as json
import pandas as pd
import yaml
from pandas import DataFrame, Series

from cbrkit import model

__all__ = ("load_path", "load_dataframe")


class DataFrameCasebase(abc.Mapping[model.CaseName, model.CaseType]):
    df: DataFrame

    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def __getitem__(self, key: int) -> Series:
        return self.df.iloc[key]

    def __iter__(self) -> Iterator[Hashable]:
        return iter(self.df.index)

    def __len__(self) -> int:
        return len(self.df)


def load_dataframe(df: DataFrame) -> model.Casebase[pd.Series]:
    return DataFrameCasebase(df)


def _load_csv(path: model.FilePath) -> dict[str, dict[str, str]]:
    data: dict[str, dict[str, str]] = {}

    with open(path) as fp:
        reader = csv.DictReader(fp)
        row: dict[str, str]

        for idx, row in enumerate(reader):
            data[str(idx)] = row

        return data


def _load_json(path: model.FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return json.loads(fp.read())


def _load_toml(path: model.FilePath) -> dict[str, Any]:
    with open(path, "rb") as fp:
        return tomllib.load(fp)


def _load_yaml(path: model.FilePath) -> dict[str, Any]:
    data: dict[str, Any] = {}

    with open(path, "rb") as fp:
        for doc in yaml.safe_load_all(fp):
            data |= doc

    return data


FileLoader = Callable[[model.FilePath], dict[str, Any]]

_file_loaders: dict[str, FileLoader] = {
    ".json": _load_json,
    ".toml": _load_toml,
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
    ".csv": _load_csv,
}


def load_path(path: model.FilePath) -> model.Casebase[Any]:
    if isinstance(path, str):
        path = Path(path)

    cb: model.Casebase[Any] | None = None

    if path.is_file():
        cb = load_file(path)
    elif path.is_dir():
        cb = load_folder(path)
    else:
        raise FileNotFoundError(path)

    if cb is None:
        raise NotImplementedError()

    return cb


def load_file(path: Path) -> model.Casebase[Any] | None:
    if path.suffix not in _file_loaders:
        return None

    loader = _file_loaders[path.suffix]

    return cast(model.Casebase[Any], loader(path))


def load_folder(path: Path, pattern: str = "**/*") -> model.Casebase[Any] | None:
    cb: model.Casebase[Any] = {}

    for file in path.glob(pattern):
        if file.is_file() and file.suffix in _file_loaders:
            loader = _file_loaders[path.suffix]
            cb[file.name] = loader(file)

    if len(cb) == 0:
        return None

    return cb
