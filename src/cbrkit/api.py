from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "api"):
    from fastapi import FastAPI, UploadFile
    from fastapi.openapi.utils import get_openapi
    from pydantic_settings import BaseSettings, SettingsConfigDict

type CasebaseSpec = dict[str, Any] | UploadFile | Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="cbrkit_")
    retriever: str = ""
    reuser: str = ""
    synthesizer: str = ""


settings = Settings()
app = FastAPI()


def load_callables(value: str) -> dict[str, Any]:
    if value == "":
        return {}

    return cbrkit.helpers.load_callables_map(value.split(","))


loaded_retrievers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc]
] = load_callables(settings.retriever)

loaded_reusers: dict[str, cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc]] = (
    load_callables(settings.reuser)
)

loaded_synthesizers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.SynthesizerFunc]
] = load_callables(settings.synthesizer)


def parse_dataset(obj: CasebaseSpec) -> Mapping[str, Any]:
    if isinstance(obj, dict):
        return obj

    data: Mapping[Any, Any] = {}

    if isinstance(obj, UploadFile):
        assert obj.content_type is not None
        loader = cbrkit.loaders.structured_loaders[f".{obj.content_type}"]
        data = loader(obj.file)
    elif isinstance(obj, Path):
        data = cbrkit.loaders.path(obj)

    if not all(isinstance(key, str) for key in data.keys()):
        return {str(key): value for key, value in data.items()}

    return data


def parse_callables[T](
    keys: list[str] | str | None,
    loaded: dict[str, T],
) -> list[T]:
    if not keys:
        keys = list(loaded.keys())
    elif isinstance(keys, str):
        keys = [keys]

    return [loaded[name] for name in keys]


class RetrieveRequest(BaseModel):
    casebase: CasebaseSpec
    queries: CasebaseSpec
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/retrieve")
def retrieve(
    request: RetrieveRequest,
) -> cbrkit.retrieval.Result[str, str, Any, cbrkit.typing.Float]:
    return cbrkit.retrieval.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
    )


class ReuseRequest(BaseModel):
    casebase: CasebaseSpec
    queries: CasebaseSpec
    reusers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/reuse")
def reuse(
    request: ReuseRequest,
) -> cbrkit.reuse.Result[str, str, Any, cbrkit.typing.Float]:
    return cbrkit.reuse.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.reusers, loaded_reusers),
    )


class CycleRequest(BaseModel):
    casebase: CasebaseSpec
    queries: CasebaseSpec
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]
    reusers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/cycle")
def cycle(
    request: CycleRequest,
) -> cbrkit.cycle.Result[str, str, Any, cbrkit.typing.Float]:
    return cbrkit.cycle.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
        parse_callables(request.reusers, loaded_reusers),
    )


class SynthesizeRequest(BaseModel):
    casebase: CasebaseSpec
    queries: CasebaseSpec
    synthesizer: str
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]
    reusers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/synthesize")
def synthesize(
    request: SynthesizeRequest,
) -> cbrkit.synthesis.Result[str, Any]:
    cycle_result = cbrkit.cycle.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
        parse_callables(request.reusers, loaded_reusers),
    )

    return cbrkit.synthesis.apply_result(
        cycle_result.final_step,
        loaded_synthesizers[request.synthesizer],
    )


def openapi_generator():
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title="CBRKit",
            version="0.1.0",
            summary="API for CBRKit",
            description="Makes it possible to perform Case-Based Reasoning tasks.",
            routes=app.routes,
        )

    return app.openapi_schema


app.openapi = openapi_generator
