from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

import cbrkit

with cbrkit.helpers.optional_dependencies("raise", "api"):
    from fastapi import FastAPI, UploadFile
    from fastapi.openapi.utils import get_openapi
    from pydantic_settings import BaseSettings, SettingsConfigDict


dataclass_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass(slots=True, frozen=True, config=dataclass_config)
class RetrievalResult:
    steps: list[cbrkit.retrieval.ResultStep[str, str, Any, cbrkit.typing.Float]]

    @classmethod
    def build(
        cls,
        obj: cbrkit.retrieval.Result[str, str, Any, cbrkit.typing.Float],
    ) -> "RetrievalResult":
        return cls(obj.steps)


@dataclass(slots=True, frozen=True, config=dataclass_config)
class ReuseResult:
    steps: list[cbrkit.reuse.ResultStep[str, str, Any, cbrkit.typing.Float]]

    @classmethod
    def build(
        cls,
        obj: cbrkit.reuse.Result[str, str, Any, cbrkit.typing.Float],
    ) -> "ReuseResult":
        return cls(obj.steps)


@dataclass(slots=True, frozen=True, config=dataclass_config)
class CycleResult:
    retrieval: cbrkit.retrieval.Result[str, str, Any, cbrkit.typing.Float]
    reuse: cbrkit.reuse.Result[str, str, Any, cbrkit.typing.Float]

    @classmethod
    def build(
        cls,
        obj: cbrkit.cycle.Result[str, str, Any, cbrkit.typing.Float],
    ) -> "CycleResult":
        return cls(obj.retrieval, obj.reuse)


@dataclass(slots=True, frozen=True, config=dataclass_config)
class SynthesisResult:
    steps: list[cbrkit.synthesis.ResultStep[str, Any]]

    @classmethod
    def build(
        cls,
        obj: cbrkit.synthesis.Result[str, Any],
    ) -> "SynthesisResult":
        return cls(obj.steps)


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


def parse_dataset(obj: dict[str, Any] | UploadFile | Path) -> Mapping[str, Any]:
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
    if keys is None:
        keys = list(loaded.keys())
    elif isinstance(keys, str):
        keys = [keys]

    return [loaded[name] for name in keys]


@app.post("/retrieve")
def retrieve(
    casebase: dict[str, Any] | UploadFile | Path,
    queries: dict[str, Any] | UploadFile | Path,
    retrievers: list[str] | None = None,
) -> RetrievalResult:
    result = cbrkit.retrieval.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
    )

    return RetrievalResult.build(result)


@app.post("/reuse")
def reuse(
    casebase: dict[str, Any] | UploadFile | Path,
    queries: dict[str, Any] | UploadFile | Path,
    reusers: list[str] | None = None,
) -> ReuseResult:
    result = cbrkit.reuse.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(reusers, loaded_reusers),
    )

    return ReuseResult.build(result)


@app.post("/cycle")
def cycle(
    casebase: dict[str, Any] | UploadFile | Path,
    queries: dict[str, Any] | UploadFile | Path,
    retrievers: list[str] | None = None,
    reusers: list[str] | None = None,
) -> CycleResult:
    result = cbrkit.cycle.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
        parse_callables(reusers, loaded_reusers),
    )

    return CycleResult.build(result)


@app.post("/synthesize")
def synthesize(
    casebase: dict[str, Any] | UploadFile | Path,
    queries: dict[str, Any] | UploadFile | Path,
    synthesizer: str,
    retrievers: list[str] | None = None,
    reusers: list[str] | None = None,
) -> SynthesisResult:
    cycle_result = cbrkit.cycle.apply_queries(
        parse_dataset(casebase),
        parse_dataset(queries),
        parse_callables(retrievers, loaded_retrievers),
        parse_callables(reusers, loaded_reusers),
    )

    synthesis_result = cbrkit.synthesis.apply_result(
        cycle_result.final_step,
        loaded_synthesizers[synthesizer],
    )

    return SynthesisResult.build(synthesis_result)


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
