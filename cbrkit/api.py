from typing import Any

from fastapi import FastAPI
from pydantic_settings import BaseSettings

import cbrkit


class Settings(BaseSettings):
    retrievers: str | None = None
    named_retrievers: str | None = None


settings = Settings()
app = FastAPI()

retrievers = (
    [] if settings.retrievers is None else cbrkit.import_retrievers(settings.retrievers)
)
named_retrievers = (
    {}
    if settings.named_retrievers is None
    else cbrkit.import_retrievers_map(settings.named_retrievers)
)


@app.post("/retrieve")
def all_retrievers(
    casebase: dict[str, Any], queries: dict[str, Any]
) -> dict[str, cbrkit.model.RetrievalResult]:
    return {
        query_name: cbrkit.retrieve(casebase, query, retrievers)
        for query_name, query in queries.items()
    }


@app.post("/retrieve/{retriever_name}")
def named_retriever(
    retriever_name: str, casebase: dict[str, Any], queries: dict[str, Any]
) -> dict[str, cbrkit.model.RetrievalResult]:
    return {
        query_name: cbrkit.retrieve(casebase, query, named_retrievers[retriever_name])
        for query_name, query in queries.items()
    }