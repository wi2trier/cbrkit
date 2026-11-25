"""Case base files and folders: their sources, members, companions, loading, and saving."""

from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any, Literal, Self, cast

import orjson
from pydantic import BaseModel, model_validator

from .._lazy import extra_hint
from ..errors import Refused
from ..model.graph import Graph, from_dict, from_json, to_dict
from ..schema import Model
from ..typing import Casebase, FilePath
from ._formats import (
    FORMATS,
    FormatName,
    _dump,
    _parse,
    _read,
    _serialize,
    atomic_write,
    by_suffix,
    malformed,
    sync_directory,
)

__all__ = [
    "COMPANION_SUFFIXES",
    "FOLDER_FORMATS",
    "FOLDER_SUFFIXES",
    "SUFFIXES",
    "CasebaseFormat",
    "Companion",
    "Source",
    "companion",
    "detect",
    "detect_files",
    "is_casebase",
    "key_candidates",
    "load",
    "load_files",
    "members",
    "save",
    "subject",
    "validate",
]


type CasebaseFormat = FormatName | Literal["bpmn"]
"""A file format of case bases or a folder of BPMN files."""


class Source(Model):
    """How the case base is stored.

    A folder holds one case per `json`, `bpmn`, or `txt` file, keyed by its relative path.
    A single `json` file of graphs maps keys to cbrkit graphs,
    and a single `txt` file is one case keyed by its name.

    Args:
        format: File format of the case base.
        folder: Whether the case base is a folder of files.
        graphs: Whether the cases are graphs.
        key_column: Column or field keying the rows of a record file, positions if None.
    """

    format: CasebaseFormat
    folder: bool = False
    graphs: bool = False
    key_column: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.format == "bpmn" and not self.graphs:
            raise ValueError("BPMN case bases consist of graphs")

        if self.folder and self.format not in FOLDER_FORMATS:
            raise ValueError(f"Folders hold {', '.join(FOLDER_FORMATS)} files")

        if self.graphs and self.format not in ("json", "bpmn"):
            raise ValueError("Graphs are stored as json or bpmn files")

        if self.key_column is not None and (self.folder or self.graphs):
            raise ValueError("Only the records of a single file have a key column")

        return self

    @property
    def writable(self) -> bool:
        """Whether cases can be written, false for BPMN, the formats without a serializer, and a single text file.

        A text file is one case, so only a folder can hold the cases that a change adds.

        Examples:
            >>> Source(format="csv").writable, Source(format="xml").writable
            (True, False)
            >>> Source(format="txt").writable, Source(format="txt", folder=True).writable
            (False, True)
        """
        return (
            self.format != "bpmn"
            and FORMATS[self.format].serialize is not None
            and (self.folder or self.format != "txt")
        )


type Companion = Literal["sidecar", "cache", "lock"]
"""A file beside a case base file or folder that belongs to it, see `companion`.

The sidecar holds its configuration, the cache the embeddings of its cases,
and the lock serializes the writers of a `FileStorage`.
"""

COMPANION_SUFFIXES: Mapping[Companion, str] = {
    "sidecar": ".cbrkit.json",
    "cache": ".cbrkit.sqlite",
    "lock": ".cbrkit.lock",
}
"""What the name of a case base is suffixed with to name each of its companions."""

FOLDER_FORMATS: tuple[CasebaseFormat, ...] = ("bpmn", "json", "txt")
"""The formats of the files a folder case base holds, in the order `detect` prefers them."""

FOLDER_SUFFIXES = frozenset(f".{format}" for format in FOLDER_FORMATS)
"""Suffixes of the files a folder case base holds, see `members`."""

SUFFIXES = frozenset(
    {".bpmn", *(suffix for format in FORMATS.values() for suffix in format.suffixes)}
)
"""Suffixes of case base files, compared in lower case."""


def _is_graph(data: Any) -> bool:
    return isinstance(data, Mapping) and "nodes" in data


def _member(path: Path) -> bool:
    """Whether a relative path names a visible case file of a supported format."""
    return (
        path.suffix.lower() in FOLDER_SUFFIXES
        and all(not part.startswith(".") for part in path.parts)
        and subject(path.name.lower()) is None
    )


def _member_path(key: Any) -> Path | None:
    """Read a folder key in the same form as a member returned by `load`."""
    if isinstance(key, str):
        path = Path(key)

        if not path.anchor and path.as_posix() == key and _member(path):
            return path

    return None


def _walk(folder: Path, prune: Callable[[Path], bool] | None = None) -> Iterator[Path]:
    """The files of a folder and its subfolders that may hold cases, lazily and unsorted.

    Companion files, hidden files and folders, whose names start with a dot,
    the subfolders that `prune` holds true for, and files without a suffix of `FOLDER_SUFFIXES` are left out.
    """
    for root, dirs, names in folder.walk():
        dirs[:] = [
            name
            for name in dirs
            if not name.startswith(".") and (prune is None or not prune(root / name))
        ]

        for name in names:
            path = root / name

            if _member(Path(name)) and path.is_file(follow_symlinks=False):
                yield path


def members(
    folder: FilePath,
    *,
    limit: int | None = None,
    prune: Callable[[Path], bool] | None = None,
) -> list[Path]:
    """The files of a folder case base that may hold cases, sorted.

    These are the `.json`, `.bpmn`, and `.txt` files of the folder and its subfolders,
    without companion files and hidden files and folders, whose names start with a dot.
    `load` reads those of the detected format.

    Args:
        folder: The folder case base.
        limit: Most files, the walk stops at the first one beyond it.
        prune: Whether to skip a subfolder, decided while walking, so its files are never visited.

    Raises:
        Refused: If the folder holds more than `limit` files.
    """
    walk = _walk(Path(folder), prune)
    found = list(walk if limit is None else islice(walk, limit + 1))

    if limit is not None and len(found) > limit:
        raise Refused(f"The folder holds more than {limit} files")

    return sorted(found)


def is_casebase(path: FilePath) -> bool:
    """Whether a file or folder may be a case base, the cheap test that `detect` refines.

    A file qualifies by its suffix, see `SUFFIXES`, and a folder by holding at least one of `members`,
    which the scan stops at.

    Examples:
        >>> is_casebase("data/cars.csv"), is_casebase("data/cars.csv.cbrkit.json"), is_casebase("notes.md")
        (True, False, False)
    """
    path = Path(path)

    if path.is_dir():
        return next(_walk(path), None) is not None

    return subject(path) is None and path.suffix.lower() in SUFFIXES


def _members(path: Path, source: Source) -> list[Path]:
    suffix = f".{source.format}"

    return [member for member in members(path) if member.suffix.lower() == suffix]


def _detect(
    path: Path, files: Sequence[Path] | None, read: Callable[[Path], bytes]
) -> Source:
    """Detects the source of a folder with the given member files, or of a single file, reading with `read`."""
    if files is not None:
        if not files:
            raise Refused(f"No {', '.join(FOLDER_FORMATS)} files in {path}")

        kind: CasebaseFormat = next(
            name
            for name in FOLDER_FORMATS
            if any(file.suffix.lower() == f".{name}" for file in files)
        )
        graphs = kind == "bpmn"

        if kind == "json":
            first = next(file for file in files if file.suffix.lower() == ".json")

            with malformed(first.name):
                graphs = _is_graph(orjson.loads(read(first)))

        return Source(format=kind, folder=True, graphs=graphs)

    if path.suffix.lower() == ".bpmn":
        return Source(format="bpmn", graphs=True)

    format = by_suffix(path)

    if format is None:
        raise Refused(f"Unsupported case base: {path}")

    if format == "json":
        with malformed(path.name):
            data = orjson.loads(read(path))

        first = next(iter(data.values()), None) if isinstance(data, dict) else None

        return Source(format=format, graphs=_is_graph(first))

    return Source(format=cast(CasebaseFormat, format))


def detect(path: FilePath) -> Source:
    """Detects the source descriptor of a case base file or folder.

    Graph JSON is recognized by the `nodes` of its first graph.
    Folders are scanned like by `load`, see `members`,
    and hold the first of `FOLDER_FORMATS` that they have files of.

    Raises:
        cbrkit.errors.Refused: If the case base has no supported format or its first JSON file is malformed.
    """
    path = Path(path)

    return _detect(path, members(path) if path.is_dir() else None, Path.read_bytes)


def detect_files(files: Mapping[str, bytes], *, folder: bool = False) -> Source:
    """Detects the source descriptor of a case base held in memory, see `detect`.

    Args:
        files: The content of the files by their path relative to the folder, or the one file by its name.
        folder: Whether the files are the members of a folder case base.

    Raises:
        cbrkit.errors.Refused: If the case base has no supported format or its first JSON file is malformed.
    """
    if not folder and len(files) != 1:
        raise Refused(f"A single file case base holds one file, not {len(files)}")

    def read(file: Path) -> bytes:
        return files[file.as_posix()]

    if folder:
        members = sorted(
            path for key in files if (path := _member_path(key)) is not None
        )
        return _detect(Path("."), members, read)

    return _detect(Path(next(iter(files))), None, read)


def key_candidates(casebase: Casebase[Any, Any]) -> list[str]:
    """The fields that could key the records of a case base as its `Source.key_column`, in the order of the first record.

    A candidate is a field of every record whose values are non-empty strings or integers, none of them repeated.
    It is only a suggestion, since nothing is keyed by a field unless a host sets it explicitly.

    Examples:
        >>> key_candidates({0: {"id": 1, "name": "a", "kind": "x"}, 1: {"id": 2, "name": "b", "kind": "x"}})
        ['id', 'name']
        >>> key_candidates({0: "text"}), key_candidates({})
        ([], [])
    """
    records = list(casebase.values())

    if not records or not all(isinstance(record, Mapping) for record in records):
        return []

    first = cast(Mapping[str, Any], records[0])

    def keys(field: str) -> bool:
        values = [cast(Mapping[str, Any], record).get(field) for record in records]

        return all(
            (isinstance(value, str) and value != "")
            or (isinstance(value, int) and not isinstance(value, bool))
            for value in values
        ) and len(set(values)) == len(values)

    return [field for field in first if isinstance(field, str) and keys(field)]


def _decode(name: str, data: bytes, source: Source) -> Any:
    """The case of the content of a file of a folder case base."""
    if source.format == "bpmn":
        with extra_hint("BPMN case bases"):
            from ..model.graph._bpmn import from_bpmn, parse_bpmn

        with malformed(name):
            return from_bpmn(parse_bpmn(data))

    with malformed(name):
        if source.format == "txt":
            return data.decode()

        return from_json(data) if source.graphs else orjson.loads(data)


def _graphs(name: str, cases: Mapping[Any, Any]) -> dict[Any, Any]:
    with malformed(name):
        return {key: from_dict(value) for key, value in cases.items()}


def load(
    path: FilePath, source: Source | None = None, limit: int | None = None
) -> Mapping[Any, Any]:
    """Loads a case base as described by its source, detected with `detect` if None.

    Folders are keyed by the paths of their files relative to the folder, single graph and text files by
    their name, and record files by the `key_column` or by position.
    A folder holds every file of the format in it and its subfolders,
    see `members`.
    A `limit` refuses a larger case base early, a folder before it reads any file and
    a csv, jsonl, parquet, or xlsx file after at most one case more than the limit,
    while the other formats are parsed as a whole.

    Raises:
        cbrkit.errors.Refused: If the case base is malformed in any way, whatever the format,
            or holds more than `limit` cases.
    """
    path = Path(path)
    source = source or detect(path)

    if source.folder:
        files = _members(path, source)

        if limit is not None and len(files) > limit:
            raise Refused(f"{path.name} holds more than {limit} cases")

        return {
            member.relative_to(path).as_posix(): _decode(
                member.name, member.read_bytes(), source
            )
            for member in files
        }

    if source.format == "bpmn":
        return {path.name: _decode(path.name, path.read_bytes(), source)}

    cases = _read(path, source.key_column, limit)

    return _graphs(path.name, cases) if source.graphs else cases


def load_files(
    files: Mapping[str, bytes], source: Source, limit: int | None = None
) -> Mapping[Any, Any]:
    """Loads a case base held in memory as described by its source, like `load` does a file or folder.

    Args:
        files: The content of the files by their path relative to the folder, or the one file by its name.
            A folder holds every file of the format in it, so the caller leaves out the others.
        source: How the case base is stored, see `detect_files`.
        limit: Most cases the case base may hold.

    Raises:
        cbrkit.errors.Refused: If the case base is malformed in any way, whatever the format,
            or holds more than `limit` cases.
    """
    if source.folder:
        names = sorted(
            path
            for key in files
            if (path := _member_path(key)) is not None
            and path.suffix.lower() == f".{source.format}"
        )

        if limit is not None and len(names) > limit:
            raise Refused(f"The folder holds more than {limit} cases")

        return {
            name.as_posix(): _decode(name.name, files[name.as_posix()], source)
            for name in names
        }

    if len(files) != 1:
        raise Refused(f"A single file case base holds one file, not {len(files)}")

    ((name, data),) = files.items()

    if source.format == "bpmn":
        return {name: _decode(name, data, source)}

    cases = _parse(name, data, source.key_column, limit)

    return _graphs(name, cases) if source.graphs else cases


def _plain(case: Any) -> Any:
    return to_dict(case) if isinstance(case, Graph) else case


def _files(
    path: Path,
    source: Source,
    casebase: Casebase[Any, Any],
    delete: Collection[Any] = (),
) -> dict[Path, bytes | None]:
    """The contents of the files that writing cases produces, None for a deleted file.

    A folder gets one file per written case, a single file holds every case.
    """
    if source.format == "bpmn" or not source.writable:
        raise Refused(f"{source.format} case bases are read-only")

    if not source.folder:
        plain = {key: _plain(case) for key, case in casebase.items()}

        return {path: _dump(source.format, plain, source.key_column)}

    root = path.resolve()
    files: dict[Path, bytes | None] = {}

    for key in dict.fromkeys((*casebase, *delete)):
        relative = _member_path(key)

        if relative is None or relative.suffix.lower() != f".{source.format}":
            raise Refused(
                f"Case key {key!r} must name a visible .{source.format} file in the case folder"
            )

        member = path / relative

        if not member.resolve().is_relative_to(root):
            raise Refused(f"Case key {key!r} leaves the folder through a symlink")

        files[member] = (
            _serialize(source.format, _plain(casebase[key]))
            if key in casebase
            else None
        )

    if len({member.as_posix().casefold() for member in files}) < len(files):
        raise Refused("Case keys must differ in more than their case")

    for member in files:
        if any(parent in files for parent in member.parents):
            raise Refused(f"Case file {member} has another case file as its parent")

    return files


def _write(files: Mapping[Path, bytes | None]) -> None:
    folders: set[Path] = set()

    for member, data in files.items():
        if data is not None:
            atomic_write(member, data, sync_folder=False)
        elif member.exists(follow_symlinks=False):
            member.unlink()
        else:
            continue

        folders.add(member.parent)

    for parent in folders:
        sync_directory(parent)


def save(path: FilePath, source: Source, casebase: Casebase[Any, Any]) -> None:
    """Writes a case base as described by its source, the counterpart of `load`.

    Single files are replaced atomically, folders get one file per case keyed by its relative path.
    A JSON or YAML file keeps its shape: records with a `key_column` stay records,
    and without one it holds a mapping if its keys are strings, and a list of records otherwise.

    Raises:
        cbrkit.errors.Refused: For read-only case bases, see `Source.writable`,
            and for invalid folder member keys or keys that miss a position.
    """
    _write(_files(Path(path), source, casebase))


def companion(path: FilePath, kind: Companion) -> Path:
    """Path of a companion of a case base file or folder, the inverse of `subject`.

    Examples:
        >>> companion("data/cars.csv", "sidecar").as_posix()
        'data/cars.csv.cbrkit.json'
    """
    path = Path(path)

    return path.with_name(f"{path.name}{COMPANION_SUFFIXES[kind]}")


def subject(path: FilePath) -> tuple[Companion, Path] | None:
    """The kind of a companion file and the case base it belongs to, the inverse of `companion`.

    Judged by the name alone, None for any other file.

    Examples:
        >>> kind, casebase = subject("data/cars.csv.cbrkit.json")
        >>> kind, casebase.as_posix()
        ('sidecar', 'data/cars.csv')
        >>> subject("data/cars.csv"), subject(".cbrkit.json")
        (None, None)
    """
    path = Path(path)

    for kind, suffix in COMPANION_SUFFIXES.items():
        if path.name.endswith(suffix) and path.name != suffix:
            return kind, path.with_name(path.name.removesuffix(suffix))

    return None


def validate[K, V: BaseModel](
    casebase: Casebase[K, Any], model: type[V]
) -> Casebase[K, V]:
    """Validates the cases of a case base with a Pydantic model.

    Args:
        casebase: The cases to validate.
        model: Pydantic model of a case.

    Examples:
        >>> from pydantic import BaseModel, NonNegativeInt
        >>> class Car(BaseModel):
        ...     price: NonNegativeInt
        ...     manufacturer: str
        >>> validate(load("data/cars-1k.csv"), Car)[0].manufacturer
        'mercedes-benz'
    """
    return {key: model.model_validate(value) for key, value in casebase.items()}
