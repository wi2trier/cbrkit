from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, cast

from pydantic import BaseModel, Field

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "api"):
    from fastapi import FastAPI, UploadFile
    from fastapi.openapi.utils import get_openapi
    from pydantic_settings import BaseSettings, SettingsConfigDict

type CasebaseSpec = dict[str, Any] | UploadFile | Path


class Settings(BaseSettings):
    """Environment-based configuration for the CBRKit API."""

    model_config = SettingsConfigDict(env_prefix="cbrkit_")
    retriever: str = ""
    reuser: str = ""
    reviser: str = ""
    retainer: str = ""
    synthesizer: str = ""


settings = Settings()
app = FastAPI()


def load_callables(value: str) -> dict[str, Any]:
    """Load callable objects from a comma-separated string of module paths."""
    if value == "":
        return {}

    return cbrkit.helpers.load_callables_map(value.split(","))


loaded_retrievers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.RetrieverFunc[Any, Any, Any]]
] = load_callables(settings.retriever)

loaded_reusers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.ReuserFunc[Any, Any, Any]]
] = load_callables(settings.reuser)

loaded_revisers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.ReviserFunc[Any, Any, Any]]
] = load_callables(settings.reviser)

loaded_retainers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.RetainerFunc[Any, Any, Any]]
] = load_callables(settings.retainer)

loaded_synthesizers: dict[
    str, cbrkit.typing.MaybeFactory[cbrkit.typing.SynthesizerFunc[Any, Any, Any, Any]]
] = load_callables(settings.synthesizer)


def parse_dataset(obj: CasebaseSpec) -> Mapping[str, Any]:
    """Parse a casebase specification into a string-keyed mapping."""
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj)

    data: Mapping[Any, Any] = {}

    if isinstance(obj, UploadFile):
        assert obj.content_type is not None
        loader = cbrkit.loaders.structured_loaders[f".{obj.content_type}"]
        data = loader(obj.file)
    elif isinstance(obj, Path):
        data = cbrkit.loaders.file(obj)

    if not all(isinstance(key, str) for key in data.keys()):
        return {str(key): value for key, value in data.items()}

    return data


def parse_callables[T](
    keys: list[str] | str | None,
    loaded: dict[str, T],
) -> list[T]:
    """Select loaded callables by name, defaulting to all available."""
    if not keys:
        keys = list(loaded.keys())
    elif isinstance(keys, str):
        keys = [keys]

    return [loaded[name] for name in keys]


class RetrieveRequest(BaseModel):
    """Request payload for the retrieval endpoint."""

    casebase: CasebaseSpec
    queries: CasebaseSpec
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/retrieve")
def retrieve(
    request: RetrieveRequest,
) -> cbrkit.retrieval.Result[str, str, Any, cbrkit.typing.Float]:
    """Handle a retrieval request and return ranked results."""
    return cbrkit.retrieval.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
    )


class ReuseRequest(BaseModel):
    """Request payload for the reuse endpoint."""

    casebase: CasebaseSpec
    queries: CasebaseSpec
    reusers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/reuse")
def reuse(
    request: ReuseRequest,
) -> cbrkit.reuse.Result[str, str, Any, cbrkit.typing.Float]:
    """Handle a reuse request and return adapted results."""
    return cbrkit.reuse.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.reusers, loaded_reusers),
    )


class CycleRequest(BaseModel):
    """Request payload for the full CBR cycle endpoint."""

    casebase: CasebaseSpec
    queries: CasebaseSpec
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]
    reusers: Annotated[list[str] | str, Field(default_factory=list)]
    revisers: Annotated[list[str] | str, Field(default_factory=list)]
    retainers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/cycle")
def cycle(
    request: CycleRequest,
) -> cbrkit.cycle.Result[str, str, Any, cbrkit.typing.Float]:
    """Handle a full CBR cycle request and return the combined result."""
    return cbrkit.cycle.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
        parse_callables(request.reusers, loaded_reusers),
        parse_callables(request.revisers, loaded_revisers),
        parse_callables(request.retainers, loaded_retainers),
    )


class SynthesizeRequest(BaseModel):
    """Request payload for the synthesis endpoint."""

    casebase: CasebaseSpec
    queries: CasebaseSpec
    synthesizer: str
    retrievers: Annotated[list[str] | str, Field(default_factory=list)]
    reusers: Annotated[list[str] | str, Field(default_factory=list)]
    revisers: Annotated[list[str] | str, Field(default_factory=list)]
    retainers: Annotated[list[str] | str, Field(default_factory=list)]


@app.post("/synthesize")
def synthesize(
    request: SynthesizeRequest,
) -> cbrkit.synthesis.Result[str, Any]:
    """Handle a synthesis request by running a CBR cycle and synthesizing the result."""
    cycle_result = cbrkit.cycle.apply_queries(
        parse_dataset(request.casebase),
        parse_dataset(request.queries),
        parse_callables(request.retrievers, loaded_retrievers),
        parse_callables(request.reusers, loaded_reusers),
        parse_callables(request.revisers, loaded_revisers),
        parse_callables(request.retainers, loaded_retainers),
    )

    return cbrkit.synthesis.apply_result(
        cycle_result.retain,
        loaded_synthesizers[request.synthesizer],
    )


def openapi_generator():
    """Generate and cache the OpenAPI schema for the CBRKit API."""
    if not app.openapi_schema:
        app.openapi_schema = get_openapi(
            title="CBRKit",
            version="0.1.0",
            summary="API for CBRKit",
            description="Makes it possible to perform Case-Based Reasoning tasks.",
            routes=app.routes,
        )

    return app.openapi_schema


app.openapi = openapi_generator  # type: ignore[assignment]
