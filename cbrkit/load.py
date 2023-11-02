import json
from typing import cast

import pandas as pd
import tomllib
import yaml

from cbrkit import model


def load_csv(path: model.FilePath) -> model.Casebase[model.CaseType]:
    return pd.read_csv(path).to_dict(orient="index")


def load_json(path: model.FilePath) -> model.Casebase[model.CaseType]:
    with open(path, "rb") as fp:
        return json.load(fp)


def load_toml(path: model.FilePath) -> model.Casebase[model.CaseType]:
    with open(path, "rb") as fp:
        return cast(model.Casebase[model.CaseType], tomllib.load(fp))


def load_yaml(path: model.FilePath) -> model.Casebase[model.CaseType]:
    data = {}

    with open(path, "rb") as fp:
        for doc in yaml.safe_load_all(fp):
            data |= doc

    return data


_mapping: dict[model.LoadFormat, model.LoadFunc] = {
    ".csv": load_csv,
    ".json": load_json,
    ".toml": load_toml,
    ".yaml": load_yaml,
    ".yml": load_yaml,
}


def get(name: model.LoadFormat) -> model.LoadFunc:
    return _mapping[name]
