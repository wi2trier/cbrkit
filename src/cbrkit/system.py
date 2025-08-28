from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, cast

from pydantic import BaseModel

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
    R1: BaseModel | None,
    R2: BaseModel | None,
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
        config: R1,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.retriever_factory:
            raise ValueError("Retriever factory is not defined.")

        return cbrkit.retrieval.apply_query(
            self.casebase,
            query,
            self.retriever_factory(config),
        ).default_query

    def reuse(
        self,
        query: V,
        config: R2,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.reuser_factory:
            raise ValueError("Reuser factory is not defined.")

        return cbrkit.reuse.apply_query(
            self.casebase,
            query,
            self.reuser_factory(config),
        ).default_query

    def cycle(
        self,
        query: V,
        retrieve_config: R1,
        reuse_config: R2,
    ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
        if not self.retriever_factory or not self.reuser_factory:
            raise ValueError("Retriever or reuser factory is not defined.")

        return cbrkit.cycle.apply_query(
            self.casebase,
            query,
            self.retriever_factory(retrieve_config),
            self.reuser_factory(reuse_config),
        ).final_step.default_query


with cbrkit.helpers.optional_dependencies():
    from fastapi import APIRouter, Body, FastAPI

    @dataclass(slots=True, frozen=True)
    class FastAPISystem[
        K: str | int,
        V: BaseModel,
        S: Float,
        R1: BaseModel | None,
        R2: BaseModel | None,
    ]:
        system: System[K, V, S, R1, R2]

        def retrieve(
            self,
            query: Annotated[V, Body],
            config: R1,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.retrieve(query, config)

        def reuse(
            self,
            query: Annotated[V, Body],
            config: R2,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.reuse(query, config)

        def cycle(
            self,
            query: Annotated[V, Body],
            retrieve_config: R1,
            reuse_config: R2,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.cycle(query, retrieve_config, reuse_config)

        def casebase(self) -> cbrkit.typing.Casebase[K, V]:
            return self.system.casebase

        def case(self, key: K) -> V | None:
            return self.system.casebase.get(key)

    def to_fastapi[
        K: str | int,
        V: BaseModel,
        S: Float,
        R1: BaseModel | None,
        R2: BaseModel | None,
        T: APIRouter | FastAPI,
    ](system: System[K, V, S, R1, R2], app: T) -> T:
        fastapi_system = FastAPISystem(system)

        if system.retriever_factory:
            app.post("/retrieve")(fastapi_system.retrieve)

        if system.reuser_factory:
            app.post("/reuse")(fastapi_system.reuse)

        if system.retriever_factory and system.reuser_factory:
            app.post("/cycle")(fastapi_system.cycle)

        app.get("/casebase")(fastapi_system.casebase)

        app.get("/casebase/{key}")(fastapi_system.case)

        return app


with cbrkit.helpers.optional_dependencies():
    import jsonref
    from fastmcp import FastMCP
    from fastmcp.tools.tool import FunctionTool
    from fastmcp.utilities.json_schema import compress_schema

    # https://github.com/jlowin/fastmcp/pull/1427
    def dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
        return compress_schema(
            cast(
                dict[str, Any],
                jsonref.replace_refs(
                    schema,
                    jsonschema=True,
                    merge_props=True,
                    lazy_load=False,
                    proxies=False,
                ),
            )
        )

    def dereference_tool(tool: FunctionTool) -> None:
        tool.parameters = dereference_schema(tool.parameters)

        if tool.output_schema is not None:
            tool.output_schema = dereference_schema(tool.output_schema)

    @dataclass(slots=True, frozen=True)
    class FastMCPSystem[
        K: str | int,
        V: BaseModel,
        S: Float,
        R1: BaseModel | None,
        R2: BaseModel | None,
    ]:
        system: System[K, V, S, R1, R2]

        def retrieve(
            self,
            query: V,
            config: R1,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.retrieve(query, config)

        def reuse(
            self,
            query: V,
            config: R2,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.reuse(query, config)

        def cycle(
            self,
            query: V,
            retrieve_config: R1,
            reuse_config: R2,
        ) -> cbrkit.retrieval.QueryResultStep[K, V, S]:
            return self.system.cycle(query, retrieve_config, reuse_config)

        def casebase(self) -> cbrkit.typing.Casebase[K, V]:
            return self.system.casebase

        def case(self, key: K) -> V | None:
            return self.system.casebase.get(key)

    def to_fastmcp[
        K: str | int,
        V: BaseModel,
        S: Float,
        R1: BaseModel | None,
        R2: BaseModel | None,
        T: FastMCP,
    ](system: System[K, V, S, R1, R2], app: T) -> T:
        fastmcp_system = FastMCPSystem(system)

        if system.retriever_factory:
            dereference_tool(app.tool("retrieve")(fastmcp_system.retrieve))

        if system.reuser_factory:
            dereference_tool(app.tool("reuse")(fastmcp_system.reuse))

        if system.retriever_factory and system.reuser_factory:
            dereference_tool(app.tool("cycle")(fastmcp_system.cycle))

        app.resource("casebase://{key}")(fastmcp_system.case)

        return app


with cbrkit.helpers.optional_dependencies():
    from pydantic_ai.toolsets import FunctionToolset

    def to_pydantic_ai(system: System) -> FunctionToolset[None]:
        tools: list[Callable[..., Any]] = []

        if system.retriever_factory:
            tools.append(system.retrieve)

        if system.reuser_factory:
            tools.append(system.reuse)

        if system.retriever_factory and system.reuser_factory:
            tools.append(system.cycle)

        return FunctionToolset(tools)
