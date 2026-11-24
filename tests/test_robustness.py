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


def test_attribute_value_falls_back_per_pair():
    def fragile(x, y):
        if x == "boom":
            raise ValueError(x)

        return equality(x, y)

    sim = cbrkit.sim.AttributeValue(attributes={"a": fragile}, default=0.0)
    scores = sim(
        [({"a": "x"}, {"a": "x"}), ({"a": "boom"}, {"a": "x"}), ({}, {"a": "x"})]
    )

    assert [score.value for score in scores] == [1.0, 0.0, 0.0]


def test_attribute_value_skips_missing_query_attributes():
    sim = cbrkit.sim.AttributeValue(
        attributes={"a": equality, "b": equality, "c": equality},
        aggregator=cbrkit.sim.Aggregator(weights={"a": 1, "b": 3, "c": 1}),
        skip_missing_query=True,
    )
    score = sim([({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": None})])[0]

    assert score.similarities == {"a": 1.0}
    assert score.value == 1.0


def test_numbers_degenerate_parameters():
    assert cbrkit.sim.numbers.Linear(max_distance=5, min_distance=5)(0, 5) == 1.0
    assert cbrkit.sim.numbers.LinearInterval(lower=3, upper=3)(3, 3) == 1.0
    assert cbrkit.sim.numbers.Sigmoid(alpha=0.001, theta=1)(0, 1000) == 0.0

    with pytest.raises(ValueError):
        cbrkit.sim.numbers.Linear(max_distance=1, min_distance=2)


def test_taxonomy_edge_cases():
    root = cbrkit.sim.taxonomy.SerializedTaxonomyNode(name="root")
    tax = cbrkit.sim.taxonomy.TaxonomyTree.parse(root)

    for func in (
        cbrkit.sim.taxonomy.WuPalmer(),
        cbrkit.sim.taxonomy.Weights(source="auto", strategy="pessimistic"),
        cbrkit.sim.taxonomy.Paths(),
    ):
        assert func("root", "root", tax) == 1.0

    assert (
        cbrkit.sim.taxonomy.Taxonomy(taxonomy=root, default=0.0)("root", "other") == 0.0
    )

    with pytest.raises(KeyError):
        cbrkit.sim.taxonomy.Taxonomy(taxonomy=root)("root", "other")

    with pytest.raises(ValueError):
        cbrkit.sim.taxonomy.TaxonomyTree.parse(
            cbrkit.sim.taxonomy.SerializedTaxonomyNode(name="root", children=["a", "a"])
        )


def test_collections_edge_cases():
    assert cbrkit.sim.collections.Jaccard()([], []) == 1.0
    assert cbrkit.sim.strings.Ngram(n=3)("ab", "ab") == 1.0
    assert cbrkit.sim.strings.Ngram(n=3)("ab", "cd") == 0.0


def test_isolated_mapping_passes_case_first():
    calls = []

    def element_sim(x, y):
        calls.append((x, y))
        return 1.0

    cbrkit.sim.collections.IsolatedMapping(element_measure=element_sim)(
        ["case"], ["query"]
    )

    assert calls == [("case", "query")]


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


def test_embedding_cache_is_keyed_by_model(tmp_path):
    def embed(value):
        async def func(texts):
            return [np.full(2, value) for _ in texts]

        return func

    path = tmp_path / "cache.sqlite3"
    first = cbrkit.sim.embed.SQLiteStore(path, "first")
    asyncio.run(
        cbrkit.sim.embed.EmbeddingCache(embedder=embed(1.0), store=first)(["text"])
    )
    table = 'odd"name'
    second = cbrkit.sim.embed.SQLiteStore(path, "second", table)
    asyncio.run(
        cbrkit.sim.embed.EmbeddingCache(embedder=embed(2.0), store=second)(["text"])
    )

    reloaded = cbrkit.sim.embed.SQLiteStore(path, "first")
    assert reloaded["text"].tolist() == [1.0, 1.0]
    assert list(cbrkit.sim.embed.SQLiteStore(path, "second", table)) == ["text"]
    del second["text"]
    assert len(second) == 0


def test_executor_runs_chunks_of_pairs():
    sizes = []

    @cbrkit.helpers.batched
    def similarity(batches):
        sizes.append(len(batches))
        return [1.0 if case == query else 0.0 for case, query in batches]

    with ThreadPoolExecutor(2) as executor:
        retriever = cbrkit.retrieve.Retriever(
            measure=similarity, executor=executor, chunksize=1
        )
        result = cbrkit.cycle.run_query({"a": "x", "b": "y"}, "x", retriever)
        distributed = cbrkit.retrieve.Distribute(
            retriever=retriever, executor=executor
        )

        assert asyncio.run(distributed([({"a": "x"}, "x")])) == [
            ({"a": "x"}, {"a": 1.0})
        ]

    assert result.similarities == {"a": 1.0, "b": 0.0}
    assert sizes[:2] == [1, 1]


def test_cycle_requires_a_function():
    with pytest.raises(ValueError):
        cbrkit.cycle.run_query({"a": "x"}, "x")


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


def test_pydantic_ai_embeddings_batch_and_fit_the_model():
    embeddings = pytest.importorskip("pydantic_ai.embeddings")
    from pydantic_ai.embeddings.test import TestEmbeddingModel

    batches: list[list[str]] = []
    received: list[object] = []

    class Model(TestEmbeddingModel):
        async def embed(self, inputs, *, input_type, settings=None):
            batches.append(list(inputs))
            received.append(settings)

            return await super().embed(inputs, input_type=input_type, settings=settings)

    embed = cbrkit.sim.embed.PydanticAI(
        embeddings.Embedder(Model()),
        input_type="document",
        batch_size=2,
        settings={"extra_body": {"truncate_prompt_tokens": -1}},
        max_input_tokens=1024,
    )
    vectors = asyncio.run(embed(["a", "b", "c", "word " * 3000]))

    assert len(vectors) == 4
    assert [len(batch) for batch in batches] == [2, 2]
    assert len(batches[1][1].split()) <= 1024
    assert all(
        settings == {"extra_body": {"truncate_prompt_tokens": -1}}
        for settings in received
    )


def test_pydantic_ai_embeddings_fit_the_limit_the_model_reports():
    embeddings = pytest.importorskip("pydantic_ai.embeddings")
    from pydantic_ai.embeddings.test import TestEmbeddingModel

    sent: list[str] = []

    class Model(TestEmbeddingModel):
        async def max_input_tokens(self):
            return 1024

        async def embed(self, inputs, *, input_type, settings=None):
            sent.extend(inputs)

            return await super().embed(inputs, input_type=input_type, settings=settings)

    embed = cbrkit.sim.embed.PydanticAI(
        embeddings.Embedder(Model()), input_type="document"
    )
    asyncio.run(embed(["word " * 3000]))

    assert len(sent[0].split()) <= 1024


def test_csv_rewrite_keeps_old_rows_beside_new_ones(tmp_path):
    path = tmp_path / "cases.csv"
    path.write_text("id,price,when\n001,15000,2024-01-02\n")
    source = cbrkit.io.detect(path).model_copy(update={"key_column": "id"})
    cases = {**cbrkit.io.load(path, source), "002": {"price": 1.5, "when": "soon"}}

    cbrkit.io.save(path, source, cases)

    assert path.read_text() == "id,price,when\n001,15000,2024-01-02\n002,1.5,soon\n"


def test_embedding_cache_persists_cases_but_not_queries(tmp_path):
    async def embed(texts):
        return [np.ones(2) for _ in texts]

    path = tmp_path / "cache.sqlite"
    store = cbrkit.sim.embed.SQLiteStore(path, "model")
    measure = cbrkit.sim.embed.Embedding(
        embedder=cbrkit.sim.embed.Embedder("e"), metric=cbrkit.sim.embed.Cosine()
    )
    config = cbrkit.spec.Config(
        source=cbrkit.io.Source(format="json"), fingerprint="", measure=measure
    )
    bound = cbrkit.spec.bind(
        config, {"e": cbrkit.sim.embed.EmbeddingCache(embedder=embed, store=store)}
    )

    cbrkit.helpers.batchify(bound.measure)([("case", "secret query")])

    assert list(store) == ["case"]

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone() == ("wal",)


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


def test_sqlite_store_survives_deleted_file(tmp_path):
    path = tmp_path / "cache.sqlite"
    store = cbrkit.sim.embed.SQLiteStore(path, "model")
    store.update({"a": np.ones(2)})
    path.unlink()

    assert store.get_many(["a"]) == {}

    store.update({"a": np.ones(2)})

    assert store["a"].tolist() == [1.0, 1.0]


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
