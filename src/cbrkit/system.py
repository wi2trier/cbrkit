from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel
from typing_extensions import Any

import cbrkit
from cbrkit.helpers import produce_sequence
from cbrkit.typing import Float, MaybeSequence

__all__ = [
    "System",
    "to_fastapi",
    "to_fastmcp",
    "to_pydantic_ai",
]


@dataclass(slots=True, frozen=True)
class System[K: str | int, V: BaseModel, S: Float]:
    casebase: cbrkit.typing.Casebase[K, V]
    model: type[V]
    retriever_pipelines: Mapping[
        str, MaybeSequence[cbrkit.typing.RetrieverFunc[K, V, S]]
    ] = field(default_factory=dict)
    reuser_pipelines: Mapping[str, MaybeSequence[cbrkit.typing.ReuserFunc[K, V, S]]] = (
        field(default_factory=dict)
    )

    def get_retriever_pipeline(
        self, name: str, limit: int | None
    ) -> Sequence[cbrkit.typing.RetrieverFunc[K, V, S]]:
        retrievers = produce_sequence(self.retriever_pipelines[name])

        if limit is not None:
            *head_retrievers, tail_retriever = retrievers
            retrievers = head_retrievers + [
                cbrkit.retrieval.dropout(tail_retriever, limit=limit)
            ]

        return retrievers

    def retrieve(
        self,
        query: V,
        retriever_pipeline: str,
        limit: int | None = None,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        return cbrkit.retrieval.apply_query(
            self.casebase,
            query,
            self.get_retriever_pipeline(retriever_pipeline, limit),
        ).default_query

    def reuse(
        self,
        query: V,
        reuser_pipeline: str,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        return cbrkit.reuse.apply_query(
            self.casebase,
            query,
            self.reuser_pipelines[reuser_pipeline],
        ).default_query

    def cycle(
        self,
        query: V,
        retriever_pipeline: str,
        reuser_pipeline: str,
        limit: int | None = None,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        return cbrkit.cycle.apply_query(
            self.casebase,
            query,
            self.get_retriever_pipeline(retriever_pipeline, limit),
            self.reuser_pipelines[reuser_pipeline],
        ).final_step.default_query

    @property
    def tools(self) -> list[Callable[..., Any]]:
        res: list[Callable[..., Any]] = []

        if self.retriever_pipelines:
            res.append(self.retrieve)

        if self.reuser_pipelines:
            res.append(self.reuse)

        if self.retriever_pipelines and self.reuser_pipelines:
            res.append(self.cycle)

        return res

    def get_case(self, name: K) -> V:
        return self.casebase[name]

    def get_retriever_names(self) -> list[str]:
        return list(self.retriever_pipelines.keys())

    def get_reuser_names(self) -> list[str]:
        return list(self.reuser_pipelines.keys())

    @property
    def resources(self) -> dict[str, Callable[..., Any]]:
        return {
            "casebase/{name}": self.get_case,
            "pipelines/retrieve": self.get_retriever_names,
            "pipelines/reuse": self.get_reuser_names,
        }

    @property
    def prompts(self) -> list[Callable[..., Any]]:
        return []


with cbrkit.helpers.optional_dependencies():
    from fastapi import FastAPI

    def to_fastapi(system: System) -> FastAPI:
        app = FastAPI()

        for value in system.tools:
            app.post(f"/tool/{value.__name__}")(value)

        for key, value in system.resources.items():
            app.get(f"/resource/{key}")(value)

        for value in system.prompts:
            app.post(f"/prompt/{value.__name__}")(value)

        return app


with cbrkit.helpers.optional_dependencies():
    from fastmcp import FastMCP

    def to_fastmcp(system: System) -> FastMCP[Any]:
        app = FastMCP()

        for value in system.tools:
            app.tool(value)

        for key, value in system.resources.items():
            app.resource(f"cbrkit://{key}")(value)

        for value in system.prompts:
            app.prompt(value)

        return app


with cbrkit.helpers.optional_dependencies():
    from pydantic_ai.toolsets import FunctionToolset

    def to_pydantic_ai(system: System) -> FunctionToolset[Any]:
        return FunctionToolset(system.tools)
