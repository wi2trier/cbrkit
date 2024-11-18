from collections.abc import Mapping
from typing import Any, Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

try:
    from fastapi import FastAPI
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ModuleNotFoundError:
    print("Please install cbrkit with the [api] extra to use the REST API server.")
    raise

import cbrkit

pydantic_dataclass_kwargs = {
    "config": ConfigDict(arbitrary_types_allowed=True),
    "frozen": True,
    "slots": True,
}

RetrievalResult = dataclass(cbrkit.retrieval.Result, **pydantic_dataclass_kwargs)
ReuseResult = dataclass(cbrkit.reuse.Result, **pydantic_dataclass_kwargs)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="cbrkit_")
    retriever: str | None = None
    retriever_map: str | None = None
    reuser: str | None = None
    reuser_map: str | None = None


settings = Settings()
app = FastAPI()

retriever: list[cbrkit.typing.RetrieverFunc] = []
retriever_map: dict[str, cbrkit.typing.RetrieverFunc] = {}

if settings.retriever is not None and settings.retriever_map is not None:
    retriever = cbrkit.helpers.load_callables(settings.retriever.split(","))
    retriever_map = cbrkit.helpers.load_callables_map(settings.retriever_map.split(","))
elif settings.retriever is not None:
    retriever = cbrkit.helpers.load_callables(settings.retriever.split(","))
    retriever_map = {str(idx): retriever for idx, retriever in enumerate(retriever)}
elif settings.retriever_map is not None:
    retriever_map = cbrkit.helpers.load_callables_map(settings.retriever_map.split(","))
    retriever = list(retriever_map.values())

reuser: list[cbrkit.typing.ReuserFunc] = []
reuser_map: dict[str, cbrkit.typing.ReuserFunc] = {}

if settings.reuser is not None and settings.reuser_map is not None:
    reuser = cbrkit.helpers.load_callables(settings.reuser.split(","))
    reuser_map = cbrkit.helpers.load_callables_map(settings.reuser_map.split(","))
elif settings.reuser is not None:
    reuser = cbrkit.helpers.load_callables(settings.reuser.split(","))
    reuser_map = {str(idx): reuser for idx, reuser in enumerate(reuser)}
elif settings.reuser_map is not None:
    reuser_map = cbrkit.helpers.load_callables_map(settings.reuser_map.split(","))
    reuser = list(reuser_map.values())


@app.post("/retrieve", response_model=Mapping[str, RetrievalResult])
def all_retrievers(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[str, cbrkit.retrieval.Result]:
    return cbrkit.retrieval.mapply(
        casebase,
        queries,
        retriever,
        processes,
        parallel,
    )


@app.post("/retrieve/{retriever_name}", response_model=Mapping[str, RetrievalResult])
def named_retriever(
    retriever_name: str,
    casebase: dict[str, Any],
    queries: dict[str, Any],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[str, cbrkit.retrieval.Result]:
    return cbrkit.retrieval.mapply(
        casebase,
        queries,
        retriever_map[retriever_name],
        processes,
        parallel,
    )


@app.post("/reuse", response_model=Mapping[str, ReuseResult])
def all_reusers(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[str, cbrkit.reuse.Result]:
    return cbrkit.reuse.mapply(
        casebase,
        queries,
        retriever,
        processes,
        parallel,
    )


@app.post("/reuse/{retriever_name}", response_model=Mapping[str, ReuseResult])
def named_reuser(
    retriever_name: str,
    casebase: dict[str, Any],
    queries: dict[str, Any],
    processes: int = 1,
    parallel: Literal["queries", "casebase"] = "queries",
) -> Mapping[str, cbrkit.reuse.Result]:
    return cbrkit.reuse.mapply(
        casebase,
        queries,
        retriever_map[retriever_name],
        processes,
        parallel,
    )
