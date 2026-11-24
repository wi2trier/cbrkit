"""Command line, in process without any network."""

import orjson
import pytest

import cbrkit
from cbrkit import spec

pytest.importorskip("typer")

from typer.testing import CliRunner

from cbrkit.cli import app


def test_infer_then_retrieve(tmp_path):
    path = tmp_path / "cars.csv"
    path.write_text("price,fuel\n10,petrol\n20,diesel\n40,petrol\n")
    runner = CliRunner()

    assert runner.invoke(app, ["infer", str(path)]).exit_code == 0
    assert spec.read(cbrkit.io.companion(path, "sidecar")).source.format == "csv"

    result = runner.invoke(
        app, ["retrieve", str(path), '{"query": {"price": 19}, "limit": 1}']
    )
    assert result.exit_code == 0
    assert orjson.loads(result.stdout)["hits"][0]["key"] == 1

    failed = runner.invoke(app, ["retrieve", str(path), '{"query": {"color": "red"}}'])
    assert failed.exit_code == 1
    assert runner.invoke(app, ["retain", str(path), "{}"]).exit_code == 1
    assert runner.invoke(app, ["maintain", str(path), "{}"]).exit_code == 1


def test_serve_needs_a_token_for_write_operations(tmp_path):
    path = tmp_path / "cars.csv"
    path.write_text("price,fuel\n10,petrol\n20,diesel\n")
    config = spec.infer(cbrkit.io.load(path), cbrkit.io.detect(path))
    spec.write(
        cbrkit.io.companion(path, "sidecar"),
        config.model_copy(update={"retain": cbrkit.retain.Retainer()}),
    )

    result = CliRunner().invoke(app, ["serve", str(path)], env={"CBRKIT_TOKEN": ""})

    assert result.exit_code == 1
    assert "write operations need --token" in result.stderr
