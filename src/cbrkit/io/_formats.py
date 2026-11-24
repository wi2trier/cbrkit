"""The file formats of case bases and the reading and writing of their documents."""

import contextlib
import csv
import datetime
import io
import os
import secrets
import stat
import tomllib
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Literal
from xml.etree import ElementTree

import orjson
import polars as pl
import yaml

from .._lazy import extra_hint
from .._util import keyed, sorted_keys
from ..errors import Refused
from ..typing import Casebase

__all__ = [
    "FORMATS",
    "Format",
    "FormatName",
    "Polars",
    "atomic_write",
    "by_suffix",
    "durable_rename",
    "is_partial",
    "malformed",
    "patch_records",
    "sync_directory",
]


@dataclass(slots=True, frozen=True)
class Polars(Mapping[Any, dict[str, Any]]):
    """A polars data frame as a mapping of its rows, the cases of a tabular file.

    Args:
        df: The data frame whose rows are the cases.
        key_column: Column whose values key the rows, positions if None.

    Examples:
        >>> cases = Polars(pl.DataFrame({"id": ["a", "b"], "price": [1, 2]}), key_column="id")
        >>> cases["b"]
        {'id': 'b', 'price': 2}
    """

    df: pl.DataFrame
    _: KW_ONLY
    key_column: str | None = None
    _rows: dict[Any, dict[str, Any]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_rows", keyed(self.df.iter_rows(named=True), self.key_column)
        )

    def __getitem__(self, key: Any) -> dict[str, Any]:
        return self._rows[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def _csv(data: bytes, rows: int | None = None) -> pl.DataFrame:
    r"""The rows of a CSV file, typed, but an integer column whose numbers would not write back its text stays text.

    An empty file, which `_csv_rows` writes for no rows, has no rows.

    Examples:
        >>> _csv(b"zip,price\n007,20.50\n").row(0)
        ('007', 20.5)
        >>> _csv(b"\n").height
        0
    """
    if not data.strip():
        return pl.DataFrame()

    typed = pl.read_csv(data, try_parse_dates=True, n_rows=rows)
    integers = [name for name, dtype in typed.schema.items() if dtype.is_integer()]

    if not integers:
        return typed

    text = pl.read_csv(data, columns=integers, infer_schema=False, n_rows=rows)

    return typed.with_columns(
        text[name]
        for name in integers
        if not typed[name].cast(pl.String).eq_missing(text[name]).all()
    )


def _cell(value: Any) -> Any:
    """A value as a CSV cell that `_csv` reads back as it."""
    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()

    return value


def _csv_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    r"""Writes records as CSV, each value as it is, so a new row neither retypes nor fails the column of an old one.

    Examples:
        >>> _csv_rows([{"id": "007", "price": 15000}, {"id": "008", "price": 1.5, "on": True}])
        b'id,price,on\n007,15000,\n008,1.5,true\n'
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows({k: _cell(v) for k, v in row.items()} for row in rows)

    return buffer.getvalue().encode()


def _csv_line(cells: Sequence[Any], ending: str) -> str:
    """One CSV record of the cells, each written as `_cell` does."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator=ending).writerow(_cell(cell) for cell in cells)

    return buffer.getvalue()


def _csv_spans(lines: Sequence[str]) -> list[str]:
    """The verbatim text of every record of CSV lines, a quoted line break kept within its record, blank lines left out."""
    reader = csv.reader(lines)
    spans: list[str] = []
    start = 0

    for row in reader:
        end = reader.line_num

        if row:
            spans.append("".join(lines[start:end]))

        start = end

    return spans


PATCHABLE: frozenset[str] = frozenset({"csv", "jsonl"})
"""The formats whose untouched records `patch_records` keeps verbatim."""


def patch_records(
    name: str,
    original: bytes,
    stored: Sequence[Any],
    cases: Mapping[Any, Any],
    changed: Collection[Any],
    key_column: str | None = None,
) -> bytes | None:
    r"""The content of a CSV or JSON Lines file with only the changed records written anew, the others kept verbatim.

    The records of the file are the stored cases in file order, as reading it gives them,
    so a record of a case that is neither changed nor removed keeps its text, such as `1.50`,
    while removed cases leave the file and new ones follow the kept ones.
    New CSV columns are appended to the header and left empty in the kept records.

    Args:
        name: The format of the file, see `FormatName`.
        original: The current content of the file.
        stored: The keys of the cases the file holds, in file order.
        cases: Every case the file holds afterwards by its key.
        changed: The keys of the replaced and added cases.
        key_column: Column or field keying the records, positions if None.

    Returns:
        The new content, None for other formats and for a file whose records are not the stored ones,
        which then has to be written whole.

    Raises:
        cbrkit.errors.Refused: If cases keyed by position miss a position.

    Examples:
        >>> patch_records("csv", b"id,price\n1,1.50\n2,2.00\n", [0, 1], {0: {"id": 1, "price": 1.5}, 1: {"id": 2, "price": 3.0}}, [1])
        b'id,price\n1,1.50\n2,3.0\n'
        >>> patch_records("jsonl", b'{"a": 1.50}\n', [0], {0: {"a": 1.5}, 1: {"a": 2}}, [1])
        b'{"a": 1.50}\n{"a":2}\n'
    """
    if name not in PATCHABLE:
        return None

    try:
        text = original.decode()
    except UnicodeDecodeError:
        return None

    # Readers drop a byte order mark, so it must not become part of the first header or record.
    bom = "\ufeff" if text.startswith("\ufeff") else ""
    lines = list(io.StringIO(text.removeprefix(bom), newline=""))
    ending = "\r\n" if lines and lines[0].endswith("\r\n") else "\n"
    spans = (
        _csv_spans(lines) if name == "csv" else [line for line in lines if line.strip()]
    )
    header = spans.pop(0) if name == "csv" and spans else None

    if len(spans) != len(stored):
        return None

    kept = {
        key: span for key, span in zip(stored, spans, strict=True) if key not in changed
    }
    known = set(stored)
    order = [key for key in stored if key in cases] + sorted_keys(
        key for key in cases if key not in known
    )

    if key_column is None:
        _check_positions(order)

    def record(key: Any) -> Any:
        return _record(cases, key, key_column)

    def terminated(span: str) -> str:
        return span if span.endswith(("\n", "\r")) else span + ending

    if name == "jsonl":
        return "".join(
            [
                bom,
                *(
                    terminated(kept[key])
                    if key in kept
                    else orjson.dumps(record(key)).decode() + ending
                    for key in order
                ),
            ]
        ).encode()

    columns = next(csv.reader(io.StringIO(header or "", newline="")), [])
    added = list(
        dict.fromkeys(
            column
            for key in order
            if key not in kept
            for column in record(key)
            if column not in columns
        )
    )

    def extended(span: str, cells: Sequence[Any]) -> str:
        if not cells:
            return terminated(span)

        body = span.rstrip("\r\n")

        return body + "," + _csv_line(cells, ending)

    rows = [
        extended(kept[key], [""] * len(added))
        if key in kept
        else _csv_line(
            [record(key).get(column) for column in [*columns, *added]], ending
        )
        for key in order
    ]
    head = extended(header, added) if header is not None else _csv_line(added, ending)

    return (bom + head + "".join(rows)).encode()


def _jsonl(data: bytes, rows: int | None = None) -> list[Any]:
    """The records of a JSONL file, of which at most `rows` are read."""
    lines = (line for line in io.BytesIO(data) if line.strip())

    return [orjson.loads(line) for line in islice(lines, rows)]


def _yaml(data: bytes) -> Any:
    """The documents of a YAML file, merged into one mapping or list."""
    documents = [doc for doc in yaml.safe_load_all(data) if doc is not None]

    if all(isinstance(doc, list) for doc in documents):
        return [item for doc in documents for item in doc]

    merged: dict[Any, Any] = {}

    for doc in documents:
        if not isinstance(doc, Mapping):
            raise Refused("YAML documents mix mappings and lists")

        if duplicates := merged.keys() & doc.keys():
            raise Refused(f"Duplicate keys across YAML documents: {duplicates}")

        merged |= doc

    return merged


def _toml(data: bytes) -> Any:
    r"""The tables of a TOML file as a mapping of cases, or the records of its only array.

    Examples:
        >>> _toml(b'[golf]\nprice = 20000')
        {'golf': {'price': 20000}}
        >>> _toml(b'[[cars]]\nname = "Golf"\n[[cars]]\nname = "Polo"')
        [{'name': 'Golf'}, {'name': 'Polo'}]
    """
    document = tomllib.loads(data.decode())

    match list(document.values()):
        case [list() as records]:
            return records
        case _:
            return document


def _scalar(text: str) -> Any:
    """A JSON number or boolean as such, any other text as it is."""
    with contextlib.suppress(orjson.JSONDecodeError):
        if isinstance(value := orjson.loads(text), bool | int | float):
            return value

    return text


def _element(element: ElementTree.Element) -> Any:
    """An element as its attributes and children by tag, repeated tags as lists, a leaf as its text."""
    children: dict[str, list[Any]] = {}

    for child in element:
        children.setdefault(child.tag, []).append(_element(child))

    if fields := {key: _scalar(value) for key, value in element.attrib.items()} | {
        tag: values[0] if len(values) == 1 else values
        for tag, values in children.items()
    }:
        return fields

    return _scalar(text) if (text := (element.text or "").strip()) else None


def _xml(data: bytes) -> list[Any]:
    """The children of the root element of an XML file as records.

    Text in elements with attributes or children is left out.
    ElementTree resolves no external entities, and the bundled expat bounds entity expansion.

    Examples:
        >>> _xml(b'<cars><car id="a"><price>2.5</price><tag>x</tag><tag>y</tag></car><car/></cars>')
        [{'id': 'a', 'price': 2.5, 'tag': ['x', 'y']}, None]
    """
    return [_element(child) for child in ElementTree.fromstring(data)]


def _parquet(rows: Any) -> bytes:
    buffer = io.BytesIO()
    pl.DataFrame(rows).write_parquet(buffer)

    return buffer.getvalue()


def _xlsx(data: bytes, rows: int | None = None) -> pl.DataFrame:
    with extra_hint("The xlsx format"):
        from fastexcel import FastExcelError

    try:
        return pl.read_excel(
            data, read_options=None if rows is None else {"n_rows": rows}
        )
    except FastExcelError as exc:
        raise ValueError(str(exc)) from exc


def _text(case: Any) -> bytes:
    if not isinstance(case, str):
        raise Refused(f"A txt case is a text, not {type(case).__name__}")

    return case.encode()


def _whole[T](parse: Callable[[bytes], T]) -> Callable[[bytes, int | None], T]:
    """A parser that cannot stop early, so it reads every row whatever the limit."""
    return lambda data, rows=None: parse(data)


@dataclass(slots=True, frozen=True)
class Format:
    """A file format of case bases.

    Args:
        suffixes: File suffixes of the format, the first one is canonical.
        parse: Reads the document of a file, of which at most the given number of rows if it can stop early.
        serialize: Writes a document, None for read-only formats.
        mappings: Whether a document may be a mapping, otherwise it is a list of records.
    """

    suffixes: tuple[str, ...]
    parse: Callable[[bytes, int | None], Any]
    serialize: Callable[[Any], bytes] | None = None
    mappings: bool = False


type FormatName = Literal[
    "csv", "json", "jsonl", "parquet", "toml", "txt", "xlsx", "xml", "yaml"
]
"""The name of a file format of case bases."""

FORMATS: Mapping[FormatName, Format] = {
    "csv": Format(
        (".csv",),
        _csv,
        _csv_rows,
    ),
    "json": Format(
        (".json",),
        _whole(orjson.loads),
        lambda document: orjson.dumps(document, option=orjson.OPT_INDENT_2),
        mappings=True,
    ),
    "jsonl": Format(
        (".jsonl", ".ndjson"),
        _jsonl,
        lambda rows: b"".join(orjson.dumps(row) + b"\n" for row in rows),
    ),
    "parquet": Format(
        (".parquet",),
        lambda data, rows=None: pl.read_parquet(data, n_rows=rows),
        _parquet,
    ),
    "toml": Format((".toml",), _whole(_toml), mappings=True),
    "txt": Format((".txt",), _whole(bytes.decode), _text),
    "xlsx": Format((".xlsx",), _xlsx),
    "xml": Format((".xml",), _whole(_xml)),
    "yaml": Format(
        (".yaml", ".yml"),
        _whole(_yaml),
        lambda document: yaml.safe_dump(document, sort_keys=False).encode(),
        mappings=True,
    ),
}
"""The formats of case base files by name."""


def by_suffix(path: Path) -> FormatName | None:
    """The name of the format of a file, None if no format has its suffix.

    Examples:
        >>> by_suffix(Path("cases.YML")), by_suffix(Path("cases.md"))
        ('yaml', None)
    """
    suffix = path.suffix.lower()

    return next((name for name, f in FORMATS.items() if suffix in f.suffixes), None)


def _cases(
    document: Any, key_column: str | None = None, name: str = ""
) -> Mapping[Any, Any]:
    """The cases of a document: a mapping as it is, records keyed by `key_column` or by position, and a text as the one case `name`.

    Raises:
        cbrkit.errors.Refused: If the document is a mapping or a text and a key column is given,
            since only records have one, which keeps the shape of a rewritten document.

    Examples:
        >>> _cases([{"id": "a"}, {"id": "b"}]), _cases({"x": 1}), _cases("hi", name="a.txt")
        ({0: {'id': 'a'}, 1: {'id': 'b'}}, {'x': 1}, {'a.txt': 'hi'})
    """
    if isinstance(document, pl.DataFrame):
        return Polars(document, key_column=key_column)

    if isinstance(document, str):
        if key_column is not None:
            raise Refused(f"A text has no key column {key_column!r}")

        return {name: document}

    if isinstance(document, Mapping):
        if key_column is not None:
            raise Refused(f"A mapping of cases has no key column {key_column!r}")

        return document

    if isinstance(document, Sequence) and not isinstance(document, str):
        return keyed(document, key_column)

    raise Refused(
        f"A case base holds a mapping or a list, not {type(document).__name__}"
    )


@contextmanager
def malformed(what: str) -> Generator[None]:
    """Raises `Refused` for any error that malformed input makes a parser raise, so callers handle one error type.

    Args:
        what: Names the input in the message, such as a file.

    Examples:
        >>> with malformed("a.json"):
        ...     orjson.loads(b"{")
        Traceback (most recent call last):
        cbrkit.errors.Refused: Malformed a.json: ...
    """
    try:
        yield
    except Refused:
        raise
    except (
        ValueError,
        RecursionError,
        pl.exceptions.PolarsError,
        yaml.YAMLError,
        ElementTree.ParseError,
    ) as exc:
        raise Refused(f"Malformed {what}: {exc}") from exc


def _parse(
    name: str,
    data: bytes,
    key_column: str | None = None,
    limit: int | None = None,
) -> Mapping[Any, Any]:
    """Parses the cases of the content of a file in one of the formats.

    Args:
        name: The name of the file, whose suffix tells the format.
        data: The content of the file.
        key_column: Column or field keying the records, positions if None.
        limit: Most cases the file may hold, the formats that can stop early read one more at most.

    Raises:
        cbrkit.errors.Refused: If no format has the suffix of the file, its content is malformed,
            or it holds more than `limit` cases.
    """
    suffix = Path(name).suffix
    format = by_suffix(Path(name))

    if format is None:
        raise Refused(f"Unsupported file type: {suffix}")

    with malformed(name):
        document = FORMATS[format].parse(data, None if limit is None else limit + 1)
        cases = _cases(document, key_column, name)

    if limit is not None and len(cases) > limit:
        raise Refused(f"{name} holds more than {limit} cases")

    return cases


def _read(
    path: Path, key_column: str | None = None, limit: int | None = None
) -> Mapping[Any, Any]:
    """Reads the cases of a file in one of the formats, see `_parse`."""
    if by_suffix(path) is None:
        raise Refused(f"Unsupported file type: {path.suffix}")

    return _parse(path.name, path.read_bytes(), key_column, limit)


def _record(casebase: Mapping[Any, Any], key: Any, key_column: str | None) -> Any:
    """The case of a key, with the key in the key field."""
    return casebase[key] if key_column is None else {**casebase[key], key_column: key}


def _check_positions(keys: Sequence[Any]) -> None:
    """Refuses cases keyed by position whose keys in file order are not 0 to n-1, since reading the file renumbers them."""
    if keys != list(range(len(keys))):
        raise Refused(
            "Cases keyed by position need the keys 0 to n-1,"
            " so removing or merging them needs a key column"
        )


def _records(casebase: Casebase[Any, Any], key_column: str | None) -> list[Any]:
    """The cases in key order, with their keys in the key field."""
    keys = sorted_keys(casebase)

    if key_column is None:
        _check_positions(keys)

    return [_record(casebase, k, key_column) for k in keys]


def _serialize(name: FormatName, document: Any) -> bytes:
    """Writes a document in a format.

    Raises:
        cbrkit.errors.Refused: If the format is read-only.
    """
    func = FORMATS[name].serialize

    if func is None:
        raise Refused(f"{name} files are read-only")

    return func(document)


def _dump(
    name: FormatName, casebase: Casebase[Any, Any], key_column: str | None = None
) -> bytes:
    r"""Writes a case base in a format, the counterpart of `_read`.

    Records with a `key_column` are written as records in key order with their keys in that field,
    and without one a format that holds mappings writes a case base with string keys as a mapping,
    since records without a key column are keyed by their positions.

    Raises:
        cbrkit.errors.Refused: If the format is read-only, or cases keyed by position miss a position.

    Examples:
        >>> _dump("jsonl", {0: {"a": 1}, 1: {"a": 2}})
        b'{"a":1}\n{"a":2}\n'
        >>> _dump("csv", {"x": {"a": 1}}, key_column="id")
        b'a,id\n1,x\n'
    """
    if (
        key_column is None
        and FORMATS[name].mappings
        and all(isinstance(k, str) for k in casebase)
    ):
        return _serialize(name, dict(casebase))

    return _serialize(name, _records(casebase, key_column))


def sync_directory(path: Path) -> None:
    """Flushes the entries of a folder to disk, so a rename or removal in it survives a power loss.

    Does nothing where a folder cannot be opened, which is anywhere but posix.
    """
    if os.name != "posix":
        return

    folder = os.open(path, os.O_RDONLY)

    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def durable_rename(source: Path, target: Path) -> None:
    """Renames a path, replacing the target, and syncs the folders of both."""
    _ = source.replace(target)
    sync_directory(target.parent)

    if source.parent != target.parent:
        sync_directory(source.parent)


_PARTIAL_PREFIX = ".~"
_PARTIAL_SUFFIX = ".partial"
_PARTIAL_BYTES = 8


def _partial_name(token: bytes) -> str:
    """The name of the partial file `atomic_write` names by a random token."""
    return f"{_PARTIAL_PREFIX}{token.hex()}{_PARTIAL_SUFFIX}"


def is_partial(name: str) -> bool:
    """Whether a file name is exactly that of a partial file `atomic_write` left behind when interrupted.

    Examples:
        >>> is_partial(".~0123456789abcdef.partial")
        True
        >>> is_partial(".~notes.partial"), is_partial(".~0123456789ABCDEF.partial")
        (False, False)
    """
    try:
        token = bytes.fromhex(
            name.removeprefix(_PARTIAL_PREFIX).removesuffix(_PARTIAL_SUFFIX)
        )
    except ValueError:
        return False

    return len(token) == _PARTIAL_BYTES and _partial_name(token) == name


def atomic_write(
    path: Path, data: bytes | str | Iterable[bytes], *, sync_folder: bool = True
) -> None:
    """Writes a file durably and atomically through a unique sibling and a rename, so readers never see a partial file.

    The sibling has a name of constant length unrelated to the target, so every valid target name works,
    and `is_partial` recognizes one an interruption left behind.
    Creates the missing parent folders, writes text as UTF-8, and streams an iterable of chunks,
    keeps the permissions of a file it replaces whatever the umask, gives a new one the mode `open` would,
    and syncs the file and, unless `sync_folder` is false, its folder to disk.
    A symlink at `path` is replaced by the file rather than written through,
    so a write never lands outside the folder of `path`.
    A caller that writes many files into a folder syncs it once with `sync_directory`.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder, "a", "b.txt")
        ...     atomic_write(path, "hello")
        ...     path.read_text()
        'hello'
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        mode: int | None = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        mode = None

    partial = path.with_name(_partial_name(secrets.token_bytes(_PARTIAL_BYTES)))
    chunks = (
        [data.encode() if isinstance(data, str) else data]
        if isinstance(data, bytes | str)
        else data
    )

    try:
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)

        with os.fdopen(fd, "wb") as file:
            if mode is not None:
                # `open` applies the umask, which the replaced file's mode already passed
                os.fchmod(file.fileno(), mode)

            for chunk in chunks:
                _ = file.write(chunk)

            file.flush()
            os.fsync(file.fileno())

        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)

    if sync_folder:
        sync_directory(path.parent)
