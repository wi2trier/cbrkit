"""Regression tests for degenerate inputs and reported bugs."""


import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import numpy as np
import pytest
import cbrkit
from cbrkit.errors import Refused


equality = cbrkit.sim.generic.Equality()


def test_yaml_keeps_entries_across_documents(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("- a\n- b\n---\n- c\n")
    assert cbrkit.io.load(path) == {0: "a", 1: "b", 2: "c"}

    path.write_text("a: 1\n---\na: 2\n")

    with pytest.raises(Refused):
        cbrkit.io.load(path)


@pytest.mark.parametrize("format", ["csv", "json", "jsonl", "parquet", "yaml"])
def test_formats_round_trip_by_key(tmp_path, format):
    cases = {"x": {"name": "a", "price": 1}, "y": {"name": "b", "price": 2}}
    source = cbrkit.io.Source(format=format, key_column="id")
    path = tmp_path / f"cases.{format}"
    cbrkit.io.save(path, source, cases)

    loaded = cbrkit.io.load(path, cbrkit.io.Source(format=format, key_column="id"))
    expected = {key: {**case, "id": key} for key, case in cases.items()}

    assert {key: dict(case) for key, case in loaded.items()} == expected
    assert set(cbrkit.io.load(path, source)) == {"x", "y"}

    (tmp_path / "dup.jsonl").write_text('{"id": "x"}\n{"id": "x"}\n')

    with pytest.raises(Refused):
        cbrkit.io.load(
            tmp_path / "dup.jsonl", cbrkit.io.Source(format="jsonl", key_column="id")
        )


def test_folders_key_by_relative_path(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text('{"v": 1}')
    (tmp_path / "y.json").write_text('{"v": 2}')

    assert dict(cbrkit.io.load(tmp_path)) == {"a/x.json": {"v": 1}, "y.json": {"v": 2}}


def test_files_in_memory_load_like_the_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text('{"v": 1}')
    (tmp_path / "y.json").write_text('{"v": 2}')
    (tmp_path / "z.txt").write_text("ignored")
    (tmp_path / "cars.csv").write_text("id,price\n1,5\n")
    files = {"a/x.json": b'{"v": 1}', "y.json": b'{"v": 2}'}

    folder = cbrkit.io.detect_files(files, folder=True)
    single = cbrkit.io.detect_files({"cars.csv": b"id,price\n1,5\n"})

    assert folder == cbrkit.io.detect(tmp_path)
    assert single == cbrkit.io.detect(tmp_path / "cars.csv")
    assert dict(cbrkit.io.load_files(files, folder)) == dict(
        cbrkit.io.load(tmp_path, folder)
    )
    assert dict(cbrkit.io.load_files({"cars.csv": b"id,price\n1,5\n"}, single)) == dict(
        cbrkit.io.load(tmp_path / "cars.csv", single)
    )


def test_read_only_formats(tmp_path):
    (tmp_path / "cars.toml").write_text("[golf]\nprice = 20000\n")
    (tmp_path / "cars.xml").write_text("<cars><car><price>20000</price></car></cars>")

    for name, key in (("cars.toml", "golf"), ("cars.xml", 0)):
        path = tmp_path / name
        source = cbrkit.io.detect(path)
        assert not source.writable
        assert cbrkit.io.load(path, source)[key] == {"price": 20000}

        with pytest.raises(Refused):
            cbrkit.io.save(path, source, {})


def test_csv_rewrite_keeps_old_rows_beside_new_ones(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("id,price,when\n001,15000,2024-01-02\n")
    source = cbrkit.io.detect(path).model_copy(update={"key_column": "id"})
    cases = {**cbrkit.io.load(path, source), "002": {"price": 1.5, "when": "soon"}}

    cbrkit.io.save(path, source, cases)

    assert path.read_text() == "id,price,when\n001,15000,2024-01-02\n002,1.5,soon\n"


def test_atomic_write_keeps_the_mode_of_the_replaced_file(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("{}")
    path.chmod(0o644)

    cbrkit.io.atomic_write(path, "[]")

    assert path.stat().st_mode & 0o777 == 0o644

    # A new file gets what `open` would give it, not the temporary file's private mode.
    new = tmp_path / "new.json"
    cbrkit.io.atomic_write(new, "[]")
    opened = tmp_path / "opened.json"
    opened.write_text("[]")

    assert new.stat().st_mode & 0o777 == opened.stat().st_mode & 0o777


def test_atomic_write_takes_names_up_to_the_filesystem_limit(tmp_path):
    # The temporary sibling must not grow with the target, which may use all 255 bytes.
    path = tmp_path / f"{'ä' * 123}.pdf"
    assert len(path.name.encode()) == 250

    cbrkit.io.atomic_write(path, "done")

    assert path.read_text() == "done"
    assert [child.name for child in tmp_path.iterdir()] == [path.name]


def test_txt_cases_are_texts(tmp_path):
    single = tmp_path / "note.txt"
    single.write_text("hello")
    source = cbrkit.io.detect(single)

    assert cbrkit.io.load(single, source) == {"note.txt": "hello"}
    assert not source.writable

    folder = tmp_path / "notes"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text("a")
    (folder / "sub" / "b.txt").write_text("b")
    source = cbrkit.io.detect(folder)

    assert source == cbrkit.io.Source(format="txt", folder=True)
    assert source.writable
    assert cbrkit.io.load(folder, source) == {"a.txt": "a", "sub/b.txt": "b"}

    cbrkit.io.save(folder, source, {"c.txt": "c"})

    assert (folder / "c.txt").read_text() == "c"

    with pytest.raises(Refused):
        cbrkit.io.save(folder, source, {"d.txt": {"not": "text"}})


@pytest.mark.parametrize(
    "name",
    [
        "a.csv",
        "a.json",
        "a.jsonl",
        "a.parquet",
        "a.toml",
        "a.xlsx",
        "a.xml",
        "a.yaml",
        "a.txt",
    ],
)
def test_load_refuses_malformed_files_as_refused(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"a,b\n1,2,3\n" if name == "a.csv" else b"\xff{[<not valid\x00")

    with pytest.raises(Refused):
        cbrkit.io.load(path)

    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.json").write_bytes(b"{")

    with pytest.raises(Refused):
        cbrkit.io.load(folder)


def test_load_refuses_missing_key_column(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "x"}\n{"other": 1}\n')

    with pytest.raises(Refused):
        cbrkit.io.load(path, cbrkit.io.Source(format="jsonl", key_column="id"))


@pytest.mark.parametrize("format", ["csv", "jsonl", "parquet", "json"])
def test_load_refuses_beyond_the_limit(tmp_path, format):
    path = tmp_path / f"cases.{format}"
    cbrkit.io.save(
        path, cbrkit.io.Source(format=format), {i: {"n": i} for i in range(5)}
    )

    assert len(cbrkit.io.load(path, limit=5)) == 5

    with pytest.raises(Refused):
        cbrkit.io.load(path, limit=4)

    folder = tmp_path / "folder"
    folder.mkdir()

    for i in range(3):
        (folder / f"{i}.json").write_text("{")

    with pytest.raises(Refused, match="more than 2"):
        cbrkit.io.load(folder, cbrkit.io.Source(format="json", folder=True), limit=2)


def test_atomic_write_keeps_the_mode_of_a_replaced_file_despite_the_umask(
    tmp_path,
) -> None:
    import os

    from cbrkit.io import atomic_write

    path = tmp_path / "a.txt"
    path.write_text("old")
    path.chmod(0o664)
    previous = os.umask(0o022)

    try:
        atomic_write(path, "new")
    finally:
        os.umask(previous)

    assert path.stat().st_mode & 0o777 == 0o664
