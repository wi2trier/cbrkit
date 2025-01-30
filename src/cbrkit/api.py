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
    reuser: str | None = None
    synthesizer: str | None = None


settings = Settings()
app = FastAPI()


loaded_retrievers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]
] = (
    cbrkit.helpers.load_callables_map(settings.retriever.split(","))
    if settings.retriever is not None
    else {}
)

loaded_reusers: dict[str, cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = (
    cbrkit.helpers.load_callables_map(settings.reuser.split(","))
    if settings.reuser is not None
    else {}
)

loaded_synthesizers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.SynthesizerFunc]
] = (
    cbrkit.helpers.load_callables_map(settings.synthesizer.split(","))
    if settings.synthesizer is not None
    else {}
)


@app.post("/retrieve", response_model=RetrievalResult)
def retrieve(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    retriever: list[str] | str | None = None,
) -> cbrkit.retrieval.Result:
    if retriever is None:
        retriever = list(loaded_retrievers.keys())
    elif isinstance(retriever, str):
        retriever = [retriever]

    return cbrkit.retrieval.apply_queries(
        casebase,
        queries,
        [loaded_retrievers[x] for x in retriever],
    )


@app.post("/reuse", response_model=ReuseResult)
def reuse(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    reuser: list[str] | str | None = None,
) -> cbrkit.reuse.Result:
    if reuser is None:
        reuser = list(loaded_reusers.keys())
    elif isinstance(reuser, str):
        reuser = [reuser]

    return cbrkit.reuse.apply_queries(
        casebase,
        queries,
        [loaded_reusers[x] for x in reuser],
    )


@app.post("/cycle", response_model=CycleResult)
def cycle(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    retriever: list[str] | str | None = None,
    reuser: list[str] | str | None = None,
) -> cbrkit.cycle.Result:
    if retriever is None:
        retriever = list(loaded_retrievers.keys())
    elif isinstance(retriever, str):
        retriever = [retriever]

    if reuser is None:
        reuser = list(loaded_reusers.keys())
    elif isinstance(reuser, str):
        reuser = [reuser]

    return cbrkit.cycle.apply_queries(
        casebase,
        queries,
        [loaded_retrievers[x] for x in retriever],
        [loaded_reusers[x] for x in reuser],
    )


@app.post("/synthesize", response_model=SynthesisResult)
def synthesize(
    casebase: dict[str, Any],
    queries: dict[str, Any],
    synthesizer: str,
    retriever: list[str] | str | None = None,
    reuser: list[str] | str | None = None,
) -> cbrkit.synthesis.Result:
    if retriever is None:
        retriever = list(loaded_retrievers.keys())
    elif isinstance(retriever, str):
        retriever = [retriever]

    if reuser is None:
        reuser = list(loaded_reusers.keys())
    elif isinstance(reuser, str):
        reuser = [reuser]

    result = cbrkit.cycle.apply_queries(
        casebase,
        queries,
        [loaded_retrievers[x] for x in retriever],
        [loaded_reusers[x] for x in reuser],
    )

    return cbrkit.synthesis.apply_result(
        result.final_step,
        loaded_synthesizers[synthesizer],
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
