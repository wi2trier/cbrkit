import tomllib
from collections import abc
from pathlib import Path
from typing import Any, Hashable, Iterator, cast

import orjson as json
import pandas as pd
import yaml
from pandas import DataFrame, Series

from cbrkit import model

__all__ = ("load_file", "load_dataframe")


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


def _load_json(path: model.FilePath) -> model.Casebase[Any]:
    with open(path, "rb") as fp:
        return json.loads(fp.read())


def _load_toml(path: model.FilePath) -> model.Casebase[Any]:
    with open(path, "rb") as fp:
        return cast(model.Casebase, tomllib.load(fp))


def _load_yaml(path: model.FilePath) -> model.Casebase[Any]:
    data: dict[str, Any] = {}

    with open(path, "rb") as fp:
        for doc in yaml.safe_load_all(fp):
            data |= doc

    return cast(model.Casebase, data)


_mapping: dict[model.LoadFormat, model.LoadFunc[Any]] = {
    ".json": _load_json,
    ".toml": _load_toml,
    ".yaml": _load_yaml,
    ".yml": _load_yaml,
}


def load_file(path: model.FilePath) -> model.Casebase[Any]:
    if isinstance(path, str):
        path = Path(path)

    format = cast(model.LoadFormat, path.suffix)

    return _mapping[format](path)
