import json
import sys

import pytest

from decider.config import JsonFileStore
from decider._bridge.bridge import Bridge
from decider._bridge.loading import components
from decider._bridge.run import main, node_path

STEPS = """\
from decider import param, step
from decider._bridge.run import node_path

SEEN = []


def eligible(score: float) -> bool:
    SEEN.append(node_path())
    return score > 50.0


@step(output="limit")
def full_limit(limit: float, bonus: float = param(1000.0)) -> float:
    '''Adds the bonus.'''
    return (limit
            + bonus)
"""

FLOW = {"flow": "credit", "components": "credit_steps", "steps": [
    {"flow": "screen", "steps": [{"step": "eligible"}]},
    {"branch": "affordable", "when": "eligible", "modifies": ["limit"], "then": [{"step": "full_limit"}]},
    {"step": "card"},
]}
CARD = {"type": "scorecard", "name": "card", "output_name": "card_score", "variables": [{
    "type": "scored", "variable_name": "score", "default": {"value": 0}, "bins": [{"value": 5}]}]}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "credit_steps").mkdir()
    (tmp_path / "credit_steps" / "__init__.py").write_text(STEPS)
    (tmp_path / "flow.yaml").write_text(json.dumps(FLOW))
    (tmp_path / "sample_request.json").write_text(json.dumps({"score": 60.0, "limit": 100.0}))
    JsonFileStore(basepath=str(tmp_path / "configs")).create_version({"card": CARD, "params": {}})
    yield tmp_path
    sys.modules.pop("credit_steps", None)


def test_describe_reads_a_flow_file_with_its_components_located(project):
    d = Bridge().describe(str(project / "flow.yaml"))
    assert d["pipeline"] == "credit" and d["pipelines"][0]["kind"] == "flow file"
    screen = d["ir"]["children"][0]
    assert screen["path"] == "credit/screen"
    assert screen["children"][0]["python"]["file"] == str(project / "credit_steps" / "__init__.py")


def test_python_range_starts_past_the_docstring_and_ends_on_the_last_line(project):
    d = Bridge().describe(str(project / "flow.yaml"))
    full_limit = d["ir"]["children"][1]["children"][1]
    assert full_limit["path"] == "credit/affordable/full_limit"
    assert (full_limit["python"]["line"], full_limit["python"]["bodyLine"], full_limit["python"]["endLine"]) == (12, 15, 16)


def test_components_lists_functions_steps_and_config_documents(project):
    assert components(str(project / "flow.yaml")) == ["eligible", "full_limit", "card"]


def test_run_prints_the_output_of_the_sample(project, capsys):
    main([str(project / "flow.yaml")])
    out = capsys.readouterr().out
    assert "1100.0" in out and "card_score" in out


def test_node_path_names_which_node_a_shared_function_runs_for(project):
    main([str(project / "flow.yaml")])
    assert sys.modules["credit_steps"].SEEN == ["credit/screen/eligible", "credit/affordable/eligible"]


def test_node_path_is_none_outside_a_run():
    assert node_path() is None
