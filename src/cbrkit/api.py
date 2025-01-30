from collections.abc import Mapping
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "api"):
    from fastapi import FastAPI, UploadFile
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


def load_callables(value: str | None) -> dict[str, Any]:
    if value is None:
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


def parse_dataset(obj: dict[str, Any] | UploadFile) -> Mapping[str, Any]:
    if isinstance(obj, UploadFile):
        assert obj.content_type is not None
        loader = cbrkit.loaders.structured_loaders[f".{obj.content_type}"]
        data = loader(obj.file)

        if not all(isinstance(key, str) for key in data.keys()):
            return {str(key): value for key, value in loader(obj.file).items()}

        return data

    return obj


def parse_callables[T](
    keys: list[str] | str | None,
    loaded: dict[str, T],
) -> list[T]:
    if keys is None:
        keys = list(loaded.keys())
    elif isinstance(keys, str):
        keys = [keys]

    return [loaded[name] for name in keys]


@app.post("/retrieve", response_model=RetrievalResult)
def retrieve(
    casebase: dict[str, Any] | UploadFile,
    queries: dict[str, Any] | UploadFile,
    retrievers: list[str] | None = None,
) -> cbrkit.retrieval.Result:
    return cbrkit.retrieval.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
    )


@app.post("/reuse", response_model=ReuseResult)
def reuse(
    casebase: dict[str, Any] | UploadFile,
    queries: dict[str, Any] | UploadFile,
    reusers: list[str] | None = None,
) -> cbrkit.reuse.Result:
    return cbrkit.reuse.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(reusers, loaded_reusers),
    )


@app.post("/cycle", response_model=CycleResult)
def cycle(
    casebase: dict[str, Any] | UploadFile,
    queries: dict[str, Any] | UploadFile,
    retrievers: list[str] | None = None,
    reusers: list[str] | None = None,
) -> cbrkit.cycle.Result:
    return cbrkit.cycle.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
        parse_callables(reusers, loaded_reusers),
    )


@app.post("/synthesize", response_model=SynthesisResult)
def synthesize(
    casebase: dict[str, Any] | UploadFile,
    queries: dict[str, Any] | UploadFile,
    synthesizer: str,
    retrievers: list[str] | None = None,
    reusers: list[str] | None = None,
) -> cbrkit.synthesis.Result:
    result = cbrkit.cycle.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
        parse_callables(reusers, loaded_reusers),
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
