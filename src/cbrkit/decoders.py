import csv as csvlib
from collections.abc import Iterable

from attr import dataclass

from .typing import ConversionFunc


@dataclass(slots=True, frozen=True)
class csv_str(ConversionFunc[Iterable[str], dict[int, dict[str, str]]]):
    def __call__(self, lines: Iterable[str]) -> dict[int, dict[str, str]]:
        data: dict[int, dict[str, str]] = {}
        reader = csvlib.DictReader(lines)
        row: dict[str, str]

        for idx, row in enumerate(reader):
            data[idx] = row

        return data
