"""This module contains the Pydantic model for validating the car data."""

from pydantic import BaseModel, NonNegativeInt
from typing import Literal


class Car(BaseModel):
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
