"""Reading and writing case bases: file formats, folders of files, graphs, and file storages.

A case base is a file or folder, described by its `Source`, which `detect` derives from it.
Files are csv, json, jsonl (or ndjson), parquet, toml, txt, xlsx, xml, and yaml (or yml), recognized by their suffix,
of which toml, xlsx, and xml are read-only, see `Source.writable`,
a txt file is one case, its text, and writable only as a member of a folder,
and graphs are BPMN (with the `bpmn` extra) or cbrkit graph JSON files.
Lists of records are keyed by the values of a key column (`key_column`), which keeps the keys stable across reloads
and rejects duplicates, or by their position if there is no key column.
A format that holds mappings writes a case base with string keys as one mapping.
Folders are scanned recursively without companion files and hidden files and folders, see `members`,
and `is_casebase` is the cheap test by suffix and folder contents that hosts offer case bases by.
No key column is ever detected, `key_candidates` only suggests the fields a host may set explicitly.
`load` reads a case base and refuses a malformed one or one beyond a limit with `cbrkit.errors.Refused`,
`save` writes it, `render` gives the files that changes would write,
and `FileStorage` serves it as a storage, which a `TypedStorage` validates as typed cases.
The companions of a case base are files beside it that belong to it, see `companion` and `subject`.
`SQLQuery` reads the rows of a SQL query over an engine of the host as cases.

Modules:
- `_formats`: the file formats (`Format`, `FORMATS`, `by_suffix`, `malformed`), `Polars`, `atomic_write` with `is_partial`, `sync_directory`, and `durable_rename`.
- `_casebase`: `Source`, `detect`, `load`, `save`, `members`, `is_casebase`, `key_candidates`, the companions, and `validate`.
- `_storage`: `FileStorage` and `render`.
- `_typed`: `TypedStorage`.
- `_sql`: `SQLQuery` (requires `sql` extra).
"""

from typing import TYPE_CHECKING

from .._lazy import lazy_exports
from ._casebase import (
    COMPANION_SUFFIXES,
    FOLDER_FORMATS,
    FOLDER_SUFFIXES,
    SUFFIXES,
    CasebaseFormat,
    Companion,
    Source,
    companion,
    detect,
    detect_files,
    is_casebase,
    key_candidates,
    load,
    load_files,
    members,
    save,
    subject,
    validate,
)
from ._formats import (
    FORMATS,
    Format,
    FormatName,
    Polars,
    atomic_write,
    by_suffix,
    durable_rename,
    is_partial,
    malformed,
    sync_directory,
)
from ._storage import FileStorage, render
from ._typed import TypedStorage

if TYPE_CHECKING:
    from ._sql import SQLQuery

__all__ = [
    "COMPANION_SUFFIXES",
    "FOLDER_FORMATS",
    "FOLDER_SUFFIXES",
    "FORMATS",
    "SUFFIXES",
    "CasebaseFormat",
    "Companion",
    "FileStorage",
    "Format",
    "FormatName",
    "Polars",
    "SQLQuery",
    "Source",
    "TypedStorage",
    "atomic_write",
    "by_suffix",
    "companion",
    "detect",
    "detect_files",
    "durable_rename",
    "is_casebase",
    "is_partial",
    "key_candidates",
    "load",
    "load_files",
    "malformed",
    "members",
    "render",
    "save",
    "subject",
    "sync_directory",
    "validate",
]

lazy_exports(__name__)
