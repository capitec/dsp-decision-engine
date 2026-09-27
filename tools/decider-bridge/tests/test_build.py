"""A pipeline served the way `decider serve` finds one: `def build(...)` in the file,
its documents from the latest `configs/<version>/` next to it, instead of a module-level
`pipeline = flow(...)` with its own `PARAMS`/`SAMPLE`.
"""
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
from decider_bridge.bridge import Bridge  # noqa: E402

TABLE_DOC = {
    "type": "decision_table", "name": "bands",
    "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
    "rows": [{"lo": None, "hi": 30.0, "band": "low"}, {"lo": 30.0, "hi": None, "band": "high"}],
    "expression": {"type": "between", "variable": "score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
    "outputs": ["band"], "default": ["other"],
}


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_find_pipelines_lists_a_top_level_build(tmp_path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("from decider import flow\n\n\ndef label(score: float) -> str:\n    return str(score)\n\n\ndef build():\n    return flow(label, name='bands')\n")
    d = Bridge().describe(str(pipeline))
    assert {"name": "build", "line": 8, "kind": "build"} in d["pipelines"]
    assert d["pipeline"] == "build"


def test_a_build_with_a_configurable_step_runs_with_its_config_params_and_sample(tmp_path):
    _write(tmp_path / "pipeline.py",
           "from decider import flow, param\n"
           "from decider.steps.tables import DecisionTableConfig  # registers 'decision_table'\n\n\n"
           "def label(band: str, prefix: str = param('x-')) -> str:\n"
           "    return prefix + band\n\n\n"
           "def build(tree):\n"
           "    return flow(tree, label, name='bands')\n")
    _write(tmp_path / "configs" / "0.0.0" / "tree.json", json.dumps(TABLE_DOC))
    _write(tmp_path / "configs" / "0.0.0" / "params.json", json.dumps({"bands": {"label": {"prefix": "lo-"}}}))
    _write(tmp_path / "sample_request.json", json.dumps({"score": 25.0}))  # one record, not a list

    pipeline = str(tmp_path / "pipeline.py")
    d = Bridge().describe(pipeline)
    assert d["values"] == {"bands": {"label": {"prefix": "lo-"}}}

    r = Bridge().trace(pipeline)
    assert r["error"] is None
    assert r["data"] == [{"score": 25.0}]
    assert r["output"]["label"] == ["lo-low"]


def test_a_zero_arg_build_runs_with_its_config_params_and_sample(tmp_path):
    _write(tmp_path / "pipeline.py",
           "from decider import flow, param\n\n\n"
           "def label(score: float, prefix: str = param('x-')) -> str:\n"
           "    return f'{prefix}{score}'\n\n\n"
           "def build():\n"
           "    return flow(label, name='bands')\n")
    _write(tmp_path / "configs" / "0.0.0" / "params.json", json.dumps({"bands": {"label": {"prefix": "z-"}}}))
    _write(tmp_path / "sample_request.json", json.dumps({"score": 3.0}))

    pipeline = str(tmp_path / "pipeline.py")
    r = Bridge().trace(pipeline)
    assert r["error"] is None
    assert r["output"]["label"] == ["z-3.0"]


def test_the_modules_own_params_and_sample_still_win_over_the_config(tmp_path):
    _write(tmp_path / "pipeline.py",
           "from decider import flow, param\n\n"
           "PARAMS = {'bands': {'label': {'prefix': 'module-'}}}\n"
           "SAMPLE = [{'score': 9.0}]\n\n\n"
           "def label(score: float, prefix: str = param('x-')) -> str:\n"
           "    return f'{prefix}{score}'\n\n\n"
           "def build():\n"
           "    return flow(label, name='bands')\n")
    _write(tmp_path / "configs" / "0.0.0" / "params.json", json.dumps({"bands": {"label": {"prefix": "config-"}}}))
    _write(tmp_path / "sample_request.json", json.dumps({"score": 1.0}))

    r = Bridge().trace(str(tmp_path / "pipeline.py"))
    assert r["error"] is None
    assert r["data"] == [{"score": 9.0}]
    assert r["output"]["label"] == ["module-9.0"]


def test_a_build_missing_its_config_document_raises_a_clear_error(tmp_path):
    _write(tmp_path / "pipeline.py", "def build(tree):\n    return tree\n")
    _write(tmp_path / "configs" / "0.0.0" / "params.json", "{}")  # no tree.json in this version
    with pytest.raises(KeyError, match="'tree'"):
        Bridge().describe(str(tmp_path / "pipeline.py"))


def test_a_build_with_no_configs_dir_is_called_with_no_arguments(tmp_path):
    _write(tmp_path / "pipeline.py", "from decider import flow\n\n\ndef label(score: float) -> str:\n    return str(score)\n\n\ndef build():\n    return flow(label, name='bands')\n")
    d = Bridge().describe(str(tmp_path / "pipeline.py"))
    assert d["values"] == {}


def test_a_build_with_no_data_at_all_raises_a_clear_error(tmp_path):
    _write(tmp_path / "pipeline.py", "from decider import flow\n\n\ndef label(score: float) -> str:\n    return str(score)\n\n\ndef build():\n    return flow(label, name='bands')\n")
    with pytest.raises(ValueError, match="no data"):
        Bridge().trace(str(tmp_path / "pipeline.py"))


EXAMPLES = os.path.join(HERE, "..", "..", "..", "example_projects")


def test_describe_and_trace_the_affordability_and_credit_limit_example_projects():
    """A smoke check on two real `def build(...)` projects, `credit_core` (project 00) on the path
    exactly as their own SERVE.md documents for serving or testing them."""
    shared = os.path.join(EXAMPLES, "00-shared-credit-core", "sonnet")
    sys.path.insert(0, shared)
    try:
        for project in ("02-affordability", "07-credit-limit-management"):
            pipeline = os.path.join(EXAMPLES, project, "sonnet", "pipeline.py")
            assert Bridge().describe(pipeline)["pipeline"] == "build"
            assert Bridge().trace(pipeline)["error"] is None
    finally:
        sys.path.remove(shared)


def test_an_edited_step_of_a_build_pipeline_is_swapped_in_mid_run(tmp_path):
    pipeline = tmp_path / "pipeline.py"
    pipeline.write_text("from decider import flow\n\n\ndef double(x: float) -> float:\n    return x * 2\n\n\n"
                        "def build():\n    return flow(double, name='f')\n")
    _write(tmp_path / "configs" / "0.0.0" / "params.json", "{}")
    _write(tmp_path / "sample_request.json", json.dumps({"x": 3.0}))
    b = Bridge()
    b.start(str(pipeline), breakpoints=["f/double"])
    b.handle({"cmd": "resume"})
    pipeline.write_text(pipeline.read_text().replace("x * 2", "x * 3"))
    b.handle({"cmd": "reload_step", "path": "f/double"})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["double"].to_list() == [9.0]
