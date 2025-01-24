from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "api"):
    from fastapi import FastAPI
    from fastapi.openapi.utils import get_openapi
    from pydantic_settings import BaseSettings, SettingsConfigDict


pydantic_dataclass_kwargs: dict[str, Any] = {
    "config": ConfigDict(arbitrary_types_allowed=True),
    "frozen": True,
    "slots": True,
}

RetrievalResult = dataclass(cbrkit.retrieval.Result, **pydantic_dataclass_kwargs)
ReuseResult = dataclass(cbrkit.reuse.Result, **pydantic_dataclass_kwargs)
CycleResult = dataclass(cbrkit.cycle.Result, **pydantic_dataclass_kwargs)
SynthesisResult = dataclass(cbrkit.synthesis.Result, **pydantic_dataclass_kwargs)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="cbrkit_")
    retriever: str | None = None
    retriever_map: str | None = None
    reuser: str | None = None
    reuser_map: str | None = None
    synthesizer: str | None = None


settings = Settings()
app = FastAPI()

retriever: list[cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]] = []
retriever_map: dict[str, cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]] = {}

if settings.retriever is not None and settings.retriever_map is not None:
    retriever = cbrkit.helpers.load_callables(settings.retriever.split(","))
    retriever_map = cbrkit.helpers.load_callables_map(settings.retriever_map.split(","))
elif settings.retriever is not None:
    retriever = cbrkit.helpers.load_callables(settings.retriever.split(","))
    retriever_map = {str(idx): retriever for idx, retriever in enumerate(retriever)}
elif settings.retriever_map is not None:
    retriever_map = cbrkit.helpers.load_callables_map(settings.retriever_map.split(","))
    retriever = list(retriever_map.values())

reuser: list[cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = []
reuser_map: dict[str, cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = {}

if settings.reuser is not None and settings.reuser_map is not None:
    reuser = cbrkit.helpers.load_callables(settings.reuser.split(","))
    reuser_map = cbrkit.helpers.load_callables_map(settings.reuser_map.split(","))
elif settings.reuser is not None:
    reuser = cbrkit.helpers.load_callables(settings.reuser.split(","))
    reuser_map = {str(idx): reuser for idx, reuser in enumerate(reuser)}
elif settings.reuser_map is not None:
    reuser_map = cbrkit.helpers.load_callables_map(settings.reuser_map.split(","))
    reuser = list(reuser_map.values())

synthesizer: cbrkit.typing.MaybeFactory[cbrkit.typing.SynthesizerFunc] | None = (
    cbrkit.helpers.load_callable(settings.synthesizer)
    if settings.synthesizer is not None
    else None
)


@app.post("/retrieve", response_model=RetrievalResult)
def all_retrievers(
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.retrieval.Result:
    return cbrkit.retrieval.apply_queries(
        casebase,
        queries,
        retriever,
    )


@app.post("/retrieve/{name}", response_model=RetrievalResult)
def named_retriever(
    name: str,
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.retrieval.Result:
    return cbrkit.retrieval.apply_queries(
        casebase,
        queries,
        retriever_map[name],
    )


@app.post("/reuse", response_model=ReuseResult)
def all_reusers(
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.reuse.Result:
    return cbrkit.reuse.apply_queries(
        casebase,
        queries,
        reuser,
    )


@app.post("/reuse/{name}", response_model=ReuseResult)
def named_reuser(
    name: str,
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.reuse.Result:
    return cbrkit.reuse.apply_queries(
        casebase,
        queries,
        reuser_map[name],
    )


@app.post("/cycle", response_model=CycleResult)
def cycle(
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.cycle.Result:
    return cbrkit.cycle.apply_queries(
        casebase,
        queries,
        retriever,
        reuser,
    )


@app.post("/synthesize", response_model=SynthesisResult)
def synthesize(
    casebase: dict[str, Any],
    queries: dict[str, Any],
) -> cbrkit.synthesis.Result:
    if synthesizer is None:
        raise ValueError("Synthesis function not provided")

    result = cbrkit.cycle.apply_queries(
        casebase,
        queries,
        retriever,
        reuser,
    )

    return cbrkit.synthesis.apply_result(
        result.final_step,
        synthesizer,
    )


def openapi_generator():
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title="CBRKit",
            version="0.1.0",
            summary="API for CBRKit",
            description="API for CBRKit",
            routes=app.routes,
        )

    return app.openapi_schema


app.openapi = openapi_generator
