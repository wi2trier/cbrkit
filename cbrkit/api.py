from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

try:
    from fastapi import FastAPI
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    print("Please install cbrkit with the [api] extra to use the REST API server.")
    raise

import cbrkit

ApiResult = dataclass(
    cbrkit.retrieval.Result, config=ConfigDict(arbitrary_types_allowed=True)
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="cbrkit_")
    retriever: str | None = None
    retriever_map: str | None = None


settings = Settings()
app = FastAPI()

retriever = []
retriever_map = {}

if settings.retriever is not None and settings.retriever_map is not None:
    retriever = cbrkit.retrieval.load(settings.retriever.split(","))
    retriever_map = cbrkit.retrieval.load_map(settings.retriever_map.split(","))
elif settings.retriever is not None:
    retriever = cbrkit.retrieval.load(settings.retriever.split(","))
    retriever_map = {str(idx): retriever for idx, retriever in enumerate(retriever)}
elif settings.retriever_map is not None:
    retriever_map = cbrkit.retrieval.load_map(settings.retriever_map.split(","))
    retriever = list(retriever_map.values())


@app.post("/retrieve", response_model=dict[str, ApiResult])
def all_retrievers(
    casebase: dict[str, Any], queries: dict[str, Any]
) -> dict[str, cbrkit.retrieval.Result]:
    return {
        query_name: cbrkit.retrieval.apply(casebase, query, retriever)
        for query_name, query in queries.items()
    }


@app.post("/retrieve/{retriever_name}", response_model=dict[str, ApiResult])
def named_retriever(
    retriever_name: str, casebase: dict[str, Any], queries: dict[str, Any]
) -> dict[str, cbrkit.retrieval.Result]:
    return {
        query_name: cbrkit.retrieval.apply(
            casebase, query, retriever_map[retriever_name]
        )
        for query_name, query in queries.items()
    }
