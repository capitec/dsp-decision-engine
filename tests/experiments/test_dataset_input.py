from __future__ import annotations

import json

import polars as pl
import pytest
from pydantic import ValidationError

from decider.data import DatasetError
from decider._tooling.bridge.bridge import Bridge
from decider.experiments import Experiment, ExperimentDef, run
from decider.experiments._experiments import InputSpec


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "datasets").mkdir()
    pl.DataFrame({"income": [1000.0] * 3, "debt": [0.0, 200.0, 400.0]}).write_parquet(tmp_path / "datasets" / "loans.parquet")
    (tmp_path / "credit.py").write_text(
        "from decider import flow, param\n"
        "def ratio(income, debt):\n"
        "    return debt / income if income > 0 else 1.0\n"
        "def approved(ratio, limit=param(0.4)):\n"
        "    return ratio <= limit\n"
        "pipeline = flow(ratio, approved, name='credit')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _def(**input):
    return ExperimentDef.model_validate({"name": "d", "flow": {"entry": "credit:pipeline"}, "input": input})


def test_run_reads_a_named_dataset_from_the_project(project):
    assert run(_def(dataset="loans")).manifest.input.row_count == 3


def test_manifest_records_the_dataset_name(project):
    assert run(_def(dataset="loans")).manifest.input.dataset == "loans"


def test_manifest_records_the_path_file_name_without_a_dataset(project):
    result = run(_def(path=str(project / "datasets" / "loans.parquet")))
    assert result.manifest.input.dataset == "loans.parquet"


def test_experiment_load_finds_the_dataset_from_the_file_location(project, monkeypatch):
    (project / "experiments").mkdir()
    file = project / "experiments" / "experiment.json"
    file.write_text(_def(dataset="loans").model_dump_json())
    monkeypatch.chdir(project / "experiments")
    assert Experiment.load(file).run().manifest.input.dataset == "loans"


def test_python_built_experiment_records_the_dataset_name(project):
    frame = pl.DataFrame({"income": [1000.0], "debt": [1.0]})
    exp = Experiment("d", __import__("credit").pipeline, frame)
    exp.def_.input.dataset = "loans"
    assert exp.run().manifest.input.dataset == "loans"


def test_path_and_dataset_together_are_invalid():
    with pytest.raises(ValidationError, match="not both"):
        InputSpec(path="a.parquet", dataset="loans")


def test_unknown_dataset_lists_the_known_ones(project):
    with pytest.raises(DatasetError, match="known: loans"):
        run(_def(dataset="nope"))


def test_a_broken_sibling_dataset_does_not_stop_the_run(project):
    (project / "datasets" / "config.json").write_text('{"broken": {"type": "nope"}}')
    assert run(_def(dataset="loans")).manifest.input.row_count == 3


def test_bridge_experiment_runs_on_a_named_dataset(project):
    path = project / "experiment.json"
    path.write_text(json.dumps({"name": "d", "flow": {"entry": "credit:pipeline"},
                                "input": {"path": str(project / "missing.parquet")}}))
    result = Bridge().handle({"id": 1, "cmd": "experiment", "definition": str(path), "dataset": "loans"})
    assert result["manifest"]["input"]["dataset"] == "loans"
