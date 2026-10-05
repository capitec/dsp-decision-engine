"""decider diff: exit 0 when nothing differs, 1 when something does."""
import json

import polars as pl
from click.testing import CliRunner

from decider.cli import cli

ENTRY = "shop.pipeline:build"


def _run(*args):
    return CliRunner().invoke(cli, ["diff", *args])


def test_a_changed_revision_exits_1_and_names_the_step(repo):
    result = _run("HEAD^", "HEAD", ENTRY)
    assert result.exit_code == 1
    assert "changed credit/ratio" in result.output


def test_an_unchanged_revision_exits_0(repo):
    assert _run("HEAD", "HEAD", ENTRY).exit_code == 0


def test_data_compares_outputs_and_json_carries_both_diffs(repo):
    pl.DataFrame({"income": [1000.0], "debt": [350.0]}).write_parquet(repo / "loans.parquet")
    result = _run("HEAD^", "HEAD", ENTRY, "--data", "loans.parquet", "--format", "json")
    assert result.exit_code == 1
    doc = json.loads(result.output)
    assert doc["outputs"]["rows_changed"] == 1 and doc["structure"]["identical"] is False


def test_data_with_no_row_changes_exits_0(repo):
    pl.DataFrame({"income": [1000.0], "debt": [350.0]}).write_parquet(repo / "loans.parquet")
    assert _run("HEAD", "HEAD", ENTRY, "--data", "loans.parquet").exit_code == 0


def test_an_unknown_revision_is_a_usage_error(repo):
    result = _run("nope", "HEAD", ENTRY)
    assert result.exit_code == 1 and "did not resolve" in result.output
