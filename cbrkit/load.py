import json
from typing import cast

import pandas as pd
import tomllib
import yaml

from cbrkit import model


def from_csv(path: model.FilePath) -> model.Casebase[model.CaseType]:
    return pd.read_csv(path).to_dict(orient="index")


def from_yaml(path: model.FilePath) -> model.Casebase[model.CaseType]:
    data = {}

    with open(path, "rb") as fp:
        for doc in yaml.safe_load_all(fp):
            data |= doc

    return data


def from_toml(path: model.FilePath) -> model.Casebase[model.CaseType]:
    with open(path, "rb") as fp:
        return cast(model.Casebase[model.CaseType], tomllib.load(fp))


def from_json(path: model.FilePath) -> model.Casebase[model.CaseType]:
    with open(path, "rb") as fp:
        return json.load(fp)


_mapping: dict[model.LoadFormat, model.LoadFunc] = {
    "csv": from_csv,
    "yaml": from_yaml,
    "yml": from_yaml,
    "toml": from_toml,
    "json": from_json,
}


def get(name: model.LoadFormat) -> model.LoadFunc:
    return _mapping[name]
