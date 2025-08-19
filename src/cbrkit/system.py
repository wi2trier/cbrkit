from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel
from typing_extensions import Any

import cbrkit
from cbrkit.typing import Float, MaybeSequence

__all__ = [
    "System",
    "to_fastapi",
    "to_fastmcp",
    "to_pydantic_ai",
]


@dataclass(slots=True, frozen=True)
class System[
    K: str | int,
    V: BaseModel,
    S: Float,
    R2: BaseModel | None,
    R1: BaseModel | None,
]:
    casebase: cbrkit.typing.Casebase[K, V]
    retriever_factory: (
        Callable[[R1], MaybeSequence[cbrkit.typing.RetrieverFunc[K, V, S]]] | None
    ) = None
    reuser_factory: (
        Callable[[R2], MaybeSequence[cbrkit.typing.ReuserFunc[K, V, S]]] | None
    ) = None

    def retrieve(
        self,
        query: V,
        parameters: R1,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.retriever_factory:
            raise ValueError("Retriever factory is not defined.")

        return cbrkit.retrieval.apply_query(
            self.casebase,
            query,
            self.retriever_factory(parameters),
        ).default_query

    def reuse(
        self,
        query: V,
        parameters: R2,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.reuser_factory:
            raise ValueError("Reuser factory is not defined.")

        return cbrkit.reuse.apply_query(
            self.casebase,
            query,
            self.reuser_factory(parameters),
        ).default_query

    def cycle(
        self,
        query: V,
        retrieve_parameters: R1,
        reuse_parameters: R2,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.retriever_factory or not self.reuser_factory:
            raise ValueError("Retriever or reuser factory is not defined.")

        return cbrkit.cycle.apply_query(
            self.casebase,
            query,
            self.retriever_factory(retrieve_parameters),
            self.reuser_factory(reuse_parameters),
        ).final_step.default_query

    @property
    def tools(self) -> list[Callable[..., Any]]:
        res: list[Callable[..., Any]] = []

        if self.retriever_factory:
            res.append(self.retrieve)

        if self.reuser_factory:
            res.append(self.reuse)

        if self.retriever_factory and self.reuser_factory:
            res.append(self.cycle)

        return res

    def get_case(self, name: K) -> V:
        return self.casebase[name]

    @property
    def resources(self) -> dict[str, Callable[..., Any]]:
        return {
            "casebase://{name}": self.get_case,
        }

    @property
    def prompts(self) -> list[Callable[..., Any]]:
        return []


with cbrkit.helpers.optional_dependencies():
    from fastapi import APIRouter, FastAPI

    def to_fastapi[T: APIRouter | FastAPI](system: System, app: T) -> T:
        for value in system.tools:
            app.post(f"/tool/{value.__name__}")(value)

        for key, value in system.resources.items():
            app.get(f"/resource/{key.replace('://', '/')}")(value)

        for value in system.prompts:
            app.post(f"/prompt/{value.__name__}")(value)

        return app


with cbrkit.helpers.optional_dependencies():
    from fastmcp import FastMCP

    def to_fastmcp[T: FastMCP](system: System, app: T) -> T:
        for value in system.tools:
            app.tool(value)

        for key, value in system.resources.items():
            app.resource(key)(value)

        for value in system.prompts:
            app.prompt(value)

        return app


with cbrkit.helpers.optional_dependencies():
    from pydantic_ai.toolsets import FunctionToolset

    def to_pydantic_ai(system: System) -> FunctionToolset[None]:
        return FunctionToolset(system.tools)
