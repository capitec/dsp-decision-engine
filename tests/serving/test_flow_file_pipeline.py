"""`RequestHandler.pipeline_fn` loads a `.yaml`/`.yml`/`.json` pipeline string as a flow file."""
import asyncio
import json
import os
import sys

from click.testing import CliRunner

from decider.cli import cli
from decider.config import JsonFileStore
from decider.serving.handler import RequestHandler

COMPONENTS = """\
from decider import param


def eligible(score: float) -> bool:
    return score > 50.0


def bonus(eligible: bool, amount: float = param(10.0)) -> float:
    return amount if eligible else 0.0
"""

SCREEN_FLOW = {"flow": "screen", "steps": [{"step": "eligible"}]}
ROOT_FLOW = {"flow": "credit", "components": "components",
             "steps": [{"flow": "screen"}, {"step": "card"}, {"step": "bonus"}]}
CARD_DOC = {"type": "scorecard", "name": "card", "output_name": "card_score", "variables": [{
    "type": "scored", "variable_name": "age", "default": {"value": 0},
    "bins": [{"value": 5, "upper_bound": {"param": "young", "default": 25}},
             {"value": 10, "lower_bound": {"param": "young", "default": 25}}]}]}

RECORD = b'{"score": 60.0, "age": 20.0}'


def _write_project(tmp_path):
    (tmp_path / "components").mkdir()
    (tmp_path / "components" / "__init__.py").write_text(COMPONENTS)
    (tmp_path / "flows").mkdir()
    (tmp_path / "flows" / "screen.yaml").write_text(json.dumps(SCREEN_FLOW))
    (tmp_path / "flow.yaml").write_text(json.dumps(ROOT_FLOW))
    store = JsonFileStore(basepath=str(tmp_path / "configs"))
    store.create_version({"card": CARD_DOC, "params": {}})
    return store


def _score(handler, record=RECORD):
    handler.stage()
    handler.activate()
    response = asyncio.run(handler.process_fn(record, "application/json", "application/json"))
    return json.loads(response.content)


def test_scores_through_the_handler_in_process(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    store = _write_project(tmp_path)
    handler = RequestHandler(store, str(tmp_path / "flow.yaml"))
    body = _score(handler)
    assert (body["card_score"], body["bonus"]) == (5.0, 10.0)


def test_a_params_change_in_a_new_version_changes_the_answer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))
    store = _write_project(tmp_path)
    store.create_version({"card": CARD_DOC, "params": {"credit": {"bonus": {"amount": 20.0}}}})
    handler = RequestHandler(store, str(tmp_path / "flow.yaml"))
    handler.stage("0.1.0")
    handler.activate()
    body = json.loads(asyncio.run(handler.process_fn(RECORD, "application/json", "application/json")).content)
    assert body["bonus"] == 20.0


def test_decider_build_succeeds_with_a_flow_file_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "components", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    result = CliRunner().invoke(cli, ["build"], env={"DECIDER_API__PIPELINE": "flow.yaml"})
    assert result.exit_code == 0, result.output
    assert "built config version 0.0.0" in result.output
