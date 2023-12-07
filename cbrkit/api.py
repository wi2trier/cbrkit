from typing import Any

from fastapi import FastAPI
from pydantic_settings import BaseSettings

import cbrkit


class Settings(BaseSettings):
    retriever: str | None = None


settings = Settings()
app = FastAPI()
retrievers = []

if settings.retriever is not None:
    retrievers = cbrkit.import_retrievers(settings.retriever)


@app.post("/retrieve")
def retrieve(
    casebase: dict[str, Any], queries: dict[str, Any]
) -> dict[str, cbrkit.model.RetrievalResult]:
    return {
        query_name: cbrkit.retrieve(casebase, query, retrievers)
        for query_name, query in queries.items()
    }
