# uv run cbrkit uvicorn --search-path ./examples cars_system:api_app

from collections.abc import Mapping
from typing import Literal

import polars as pl
from fastapi import FastAPI
from fastmcp import FastMCP
from pydantic import BaseModel, NonNegativeInt

import cbrkit


class CarModel(BaseModel):
    price: NonNegativeInt
    year: NonNegativeInt
    manufacturer: str
    make: str
    fuel: Literal["gas", "diesel"]
    miles: NonNegativeInt
    title_status: Literal["clean", "rebuilt"]
    transmission: Literal["automatic", "manual"]
    drive: Literal["fwd", "rwd", "4wd"]
    type: str
    paint_color: str


class RetrieverConfig(BaseModel):
    limit: NonNegativeInt = 5


class RetrieveRequest(BaseModel):
    query: CarModel
    config: RetrieverConfig | None


type RetrieveResponse = cbrkit.retrieval.QueryResultStep[
    int, CarModel, cbrkit.sim.AttributeValueSim[float]
]


def retriever_factory(
    config: RetrieverConfig | None,
) -> cbrkit.typing.RetrieverFunc[int, CarModel, cbrkit.sim.AttributeValueSim[float]]:
    if not config:
        config = RetrieverConfig()

    return cbrkit.retrieval.dropout(
        cbrkit.retrieval.build(
            cbrkit.sim.attribute_value(
                attributes={
                    "year": cbrkit.sim.numbers.linear(max=50),
                    "make": cbrkit.sim.strings.levenshtein(),
                    "miles": cbrkit.sim.numbers.linear(max=1000000),
                },
                aggregator=cbrkit.sim.aggregator(pooling="mean"),
            ),
        ),
        limit=config.limit,
    )


df = pl.read_csv("data/cars-1k.csv")
casebase = cbrkit.loaders.validate(cbrkit.loaders.polars(df), CarModel)
system = cbrkit.system.System(
    casebase=casebase,
    retriever_factory=retriever_factory,
)

mcp = FastMCP("cbrkit")


@mcp.tool("retrieve")
def mcp_retrieve(
    query: CarModel,
    config: RetrieverConfig | None = None,
) -> RetrieveResponse:
    return system.retrieve(query, config=config)


@mcp.resource("casebase://{key}")
def mcp_case(key: int) -> CarModel:
    return cbrkit.helpers.produce_factory(system.casebase)[key]


mcp_app = mcp.http_app("/")

api_app = FastAPI(lifespan=mcp_app.lifespan)
api_app.mount("/mcp", mcp_app)


@api_app.head("/health")
def health() -> None:
    return None


@api_app.post("/retrieve")
def api_retrieve(req: RetrieveRequest) -> RetrieveResponse:
    return system.retrieve(req.query, config=req.config)


@api_app.get("/casebase")
def api_casebase() -> Mapping[int, CarModel]:
    return cbrkit.helpers.produce_factory(system.casebase)


@api_app.get("/casebase/{key}")
def api_case(key: int) -> CarModel:
    return cbrkit.helpers.produce_factory(system.casebase)[key]
