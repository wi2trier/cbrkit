"""Case base files as storages and the files that their changes write."""

import asyncio
import contextlib
import threading
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, override
from uuid import uuid7

from filelock import FileLock

from .._util import next_keys, patched, plan_patch
from ..helpers import frozendict
from ..model import Changes
from ..typing import Casebase, FilePath, Storage
from ._casebase import Source, _files, _members, _write, companion, load
from ._formats import PATCHABLE, patch_records

__all__ = ["FileStorage", "render"]


@dataclass(frozen=True, slots=True)
class _Stamp:
    """A file with its modification time and size, which change with its content."""

    path: Path
    mtime: int
    size: int


def _folder_keys(source: Source, taken: Collection[Any], count: int) -> list[str]:
    """New `<uuid>.<format>` files of a folder."""
    return [f"{uuid7()}.{source.format}" for _ in range(count)]


def _patch(
    source: Source,
    stored: Casebase[Any, Any],
    upsert: Casebase[Any, Any] | None,
    delete: Collection[Any] | None,
    add: Sequence[Any],
    insert: Casebase[Any, Any] | None,
    expected: Casebase[Any, Any] | None,
) -> tuple[dict[Any, Any], set[Any], list[Any]]:
    """The cases that a mutation writes, the keys that it deletes, and the keys of the added cases, see `plan_patch`.

    New cases get the next positions or integers of a record file, new UUIDv7 strings for string keys,
    and new `<uuid>.<format>` files in a folder.
    """
    new_keys = partial(_folder_keys, source) if source.folder else next_keys

    return plan_patch(stored, upsert, delete, add, insert, expected, new_keys)


def _contents(
    path: Path,
    source: Source,
    stored: Casebase[Any, Any],
    written: Casebase[Any, Any],
    deleted: Collection[Any],
    original: bytes | None,
) -> dict[Path, bytes | None]:
    """The files that writing and deleting cases produces, a single record file with an `original` keeping its untouched records verbatim.

    The records are written whole if the format cannot keep them, see `patch_records`,
    and for removed cases keyed by position, whose removal `_files` refuses since the positions after them would shift.
    """
    if source.folder:
        return _files(path, source, written, deleted)

    cases = patched(stored, written, deleted)

    if (
        original is not None
        and not (source.key_column is None and stored.keys() & deleted)
        and (
            data := patch_records(
                source.format,
                original,
                list(stored),
                cases,
                written.keys(),
                source.key_column,
            )
        )
        is not None
    ):
        return {path: data}

    return _files(path, source, cases)


def render(
    path: FilePath,
    source: Source,
    stored: Casebase[Any, Any],
    changes: Changes[Any, Any],
    original: bytes | None = None,
) -> dict[Path, bytes | None]:
    """The contents of the files that applying changes to a case base would write, None for a deleted file.

    Nothing is written, so a host can review the files before it saves them,
    and the result matches what `FileStorage` writes for the same changes, keys of added cases aside.

    Args:
        path: Path of the case base file or folder.
        source: How the case base is stored.
        stored: The stored cases.
        changes: The changes, such as of a retain or maintenance run.
        original: The current content of a single file case base, read as `stored`,
            whose untouched records a CSV or JSON Lines file keeps verbatim, see `patch_records`.

    Raises:
        ValueError: If the changes change a case twice.
        cbrkit.errors.Conflict: If the stored cases differ from those the changes were planned on
            or hold the key of an addition.
        cbrkit.errors.Refused: For read-only formats and invalid folder member keys.

    Examples:
        >>> from cbrkit.model import Add, Changes
        >>> changes = Changes[int, dict](items=[Add(value={"a": 2}, rule="solution", reason="new")])
        >>> render("cases.jsonl", Source(format="jsonl"), {0: {"a": 1}}, changes)
        {PosixPath('cases.jsonl'): b'{"a":1}\\n{"a":2}\\n'}
    """
    upsert, delete, add, insert = changes.patch()
    written, deleted, _ = _patch(
        source, stored, upsert, delete, add, insert, changes.expected
    )

    return _contents(Path(path), source, stored, written, deleted, original)


@dataclass(slots=True)
class _State:
    """The last loaded cases of a `FileStorage` with the stamp of their files and the lock of its threads."""

    cases: frozendict[Any, Any] = field(default_factory=frozendict)
    stamp: tuple[_Stamp, ...] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass(slots=True, frozen=True)
class FileStorage(Storage[Any, Any]):
    """The case base file or folder of a specification as a storage.

    Reading reloads the cases whenever the file or any file of the folder changed,
    for instance by another process, and returns an immutable snapshot.
    Its methods are asynchronous and run the file I/O in a worker thread.
    Writing holds the lock companion of the case base (`<name>.cbrkit.lock`), reloads the current cases,
    applies the mutation and replaces the file atomically, so concurrent writers lose no updates.
    New cases get their keys under the same lock: the next positions or integers of a record file,
    new UUIDv7 strings for string keys, and new `<uuid>.<format>` files in a folder.
    `render` gives the files of a mutation without writing them.

    Args:
        path: Path of the case base file or folder.
        source: How the case base is stored, see `detect`.

    Examples:
        >>> import asyncio
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "cases.jsonl"
        >>> storage = FileStorage(path=path, source=Source(format="jsonl"))
        >>> asyncio.run(storage.patch_index(upsert={0: {"price": 10}, 1: {"price": 20}}))
        []
        >>> path.read_text().splitlines()[1]
        '{"price":20}'
        >>> asyncio.run(storage.patch_index(add=[{"price": 30}]))
        [2]
        >>> dict(asyncio.run(FileStorage(path=path, source=Source(format="jsonl")).get_index()))[2]
        {'price': 30}
    """

    path: Path
    source: Source
    _state: _State = field(
        default_factory=_State, init=False, repr=False, compare=False
    )

    def _current(self) -> tuple[_Stamp, ...] | None:
        """The paths, modification times, and sizes of the files of the case base, None if it is missing."""
        if not self.path.exists():
            return None

        paths = _members(self.path, self.source) if self.source.folder else [self.path]
        stamps: list[_Stamp] = []

        for path in paths:
            with contextlib.suppress(FileNotFoundError):
                stat = path.stat()
                stamps.append(
                    _Stamp(path=path, mtime=stat.st_mtime_ns, size=stat.st_size)
                )

        return tuple(stamps)

    def _reload(self) -> None:
        stamp = self._current()

        if stamp != self._state.stamp:
            self._state.cases = (
                frozendict(load(self.path, self.source))
                if stamp is not None
                else frozendict()
            )
            self._state.stamp = stamp

    def _cases(self) -> Casebase[Any, Any]:
        if self._current() != self._state.stamp:
            with self._state.lock:
                self._reload()

        return self._state.cases

    @override
    async def get_index(self) -> Casebase[Any, Any]:
        """The current cases."""
        return await asyncio.to_thread(self._cases)

    @override
    async def has_index(self) -> bool:
        """Whether there are cases."""
        return bool(await self.get_index())

    def _apply(
        self, written: Casebase[Any, Any], deleted: Collection[Any], keep: bool
    ) -> None:
        """Writes and deletes cases, a single record file keeping the records of the other cases verbatim if `keep` is set."""
        original = None

        if keep and not self.source.folder and self.source.format in PATCHABLE:
            with contextlib.suppress(FileNotFoundError):
                original = self.path.read_bytes()

        stored = self._state.cases
        _write(_contents(self.path, self.source, stored, written, deleted, original))
        self._state.cases = frozendict(patched(stored, written, deleted))
        self._state.stamp = self._current()

    @override
    async def put_index(self, data: Casebase[Any, Any], /) -> None:
        """Replace all cases."""
        await asyncio.to_thread(self._put, data)

    def _put(self, data: Casebase[Any, Any]) -> None:
        with self._state.lock, FileLock(companion(self.path, "lock")):
            self._reload()
            # deleting every stored case writes the cases in the order of the data
            self._apply(data, self._state.cases.keys(), keep=False)

    @override
    async def patch_index(
        self,
        upsert: Casebase[Any, Any] | None = None,
        delete: Collection[Any] | None = None,
        add: Sequence[Any] = (),
        *,
        insert: Casebase[Any, Any] | None = None,
        expected: Casebase[Any, Any] | None = None,
    ) -> list[Any]:
        """Insert, replace, delete, and add cases as one mutation."""
        return await asyncio.to_thread(
            self._patch, upsert, delete, add, insert, expected
        )

    def _patch(
        self,
        upsert: Casebase[Any, Any] | None,
        delete: Collection[Any] | None,
        add: Sequence[Any],
        insert: Casebase[Any, Any] | None,
        expected: Casebase[Any, Any] | None,
    ) -> list[Any]:
        if not upsert and not delete and not add and not insert:
            return []

        with self._state.lock, FileLock(companion(self.path, "lock")):
            self._reload()
            written, deleted, new = _patch(
                self.source, self._state.cases, upsert, delete, add, insert, expected
            )
            self._apply(written, deleted, keep=True)

        return new
