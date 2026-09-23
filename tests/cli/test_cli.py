"""`decider template`, `build` and `serve` over a generated starter project."""
import json
import os
import sys
import types

import pytest
from click.testing import CliRunner

import decider.settings as settings_module
from decider.cli import CPU_TARGET_FILE, cli
from decider.engine.compile import cpu_target
from decider.serving.handler import construct_handler_from_settings


@pytest.fixture
def project(tmp_path, monkeypatch):
    # The CLI reads settings from the environment and imports `pipeline`/`inference` by name,
    # so each test gets a clean environment, settings object and module cache.
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIDER_")})
    monkeypatch.setattr(settings_module, "settings", settings_module.settings)
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ("pipeline", "inference"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["template", "credit-risk", "proj"])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(tmp_path / "proj")
    return tmp_path / "proj"


def test_template_writes_a_starter_project(project):
    files = sorted(p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file())
    assert files == ["README.md", "configs/0.0.0/params.json", "inference.py", "pipeline.py"]
    assert json.loads((project / "configs/0.0.0/params.json").read_text()) == {"credit_risk": {"approved": {"limit": 0.4}}}
    assert 'name="credit_risk"' in (project / "pipeline.py").read_text()


def test_template_refuses_a_non_empty_directory(project):
    result = CliRunner().invoke(cli, ["template", "again", str(project)])
    assert result.exit_code != 0
    assert "not empty" in result.output


def test_build_stages_warms_and_records_the_cpu_target(project):
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.0.0" in result.output
    assert tuple(json.loads((project / CPU_TARGET_FILE).read_text())) == cpu_target()
    assert any(p.suffix == ".nbi" for p in (project / "__pycache__").iterdir())


def test_the_built_project_serves_the_generated_handler(project):
    assert CliRunner().invoke(cli, ["build"]).exit_code == 0
    handler = construct_handler_from_settings()
    assert type(handler).__name__ == "Handler" and type(handler).__module__ == "inference"
    handler.stage()
    handler.activate()
    live = handler.module_fn()
    assert live.executable.score({"income": 1000.0, "debt": 500.0}, live.params)["approved"] is False
    assert live.executable.score({"income": 1000.0, "debt": None}, live.params)["approved"] is True


def test_build_fails_with_a_clear_message_on_an_invalid_params_document(project):
    (project / "configs/0.0.0/params.json").write_text(json.dumps({"credit_risk": {"approved": {"limit": 7.0}}}))
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code != 0
    assert "config version latest failed to build" in result.output
    assert "limit" in result.output


def test_build_fails_cleanly_on_an_unknown_version(project):
    result = CliRunner().invoke(cli, ["build", "9.9.9"])
    assert result.exit_code != 0
    assert "9.9.9 not found" in result.output


def test_build_fails_cleanly_when_the_pipeline_is_not_importable(project):
    result = CliRunner().invoke(cli, ["build"], env={"DECIDER_API__PIPELINE": "nowhere:build"})
    assert result.exit_code != 0
    assert "No module named 'nowhere'" in result.output


def test_build_honours_an_explicit_pipeline_attribute(project):
    with open(project / "pipeline.py", "a") as f:
        f.write("\n\ndef other():\n    return flow(debt_ratio, name='other')\n")
    (project / "configs/0.0.0/params.json").write_text("{}")
    result = CliRunner().invoke(cli, ["build"], env={"DECIDER_API__PIPELINE": "pipeline:other"})
    assert result.exit_code == 0, result.output
    assert "pipeline pipeline:other" in result.output


@pytest.fixture
def uvicorn_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **kw: calls.append((a, kw))))
    return calls


def test_serve_starts_the_starlette_factory_with_the_given_settings(project, uvicorn_calls):
    result = CliRunner().invoke(cli, ["serve", "--port", "9001", "--workers", "3", "--mode", "stepped"])
    assert result.exit_code == 0, result.output
    assert uvicorn_calls == [(("decider.serving.servers.starlette:create_app",),
                              {"factory": True, "host": "0.0.0.0", "port": 9001, "workers": 3})]
    # What each worker's app factory builds its handler from.
    handler = construct_handler_from_settings()
    assert (type(handler).__name__, handler.mode, handler.pipeline) == ("Handler", "stepped", "pipeline:build")
    assert os.environ["DECIDER_API__MODE"] == "stepped"


def test_serve_warns_when_the_build_ran_on_another_cpu(project, uvicorn_calls):
    triple, _, features = cpu_target()
    (project / CPU_TARGET_FILE).write_text(json.dumps([triple, "some-other-cpu", features]))
    result = CliRunner().invoke(cli, ["serve", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert "some-other-cpu" in result.output and "decider build" in result.output
    assert len(uvicorn_calls) == 1


def test_serve_is_quiet_on_the_cpu_it_was_built_for(project, uvicorn_calls):
    assert CliRunner().invoke(cli, ["build"]).exit_code == 0
    result = CliRunner().invoke(cli, ["serve", "--workers", "1"])
    assert result.exit_code == 0 and "warning" not in result.output


def test_serve_with_sanic_missing_is_a_clean_error(project, monkeypatch):
    monkeypatch.setitem(sys.modules, "sanic", None)
    result = CliRunner().invoke(cli, ["serve", "--server", "sanic"])
    assert result.exit_code != 0
    assert "pip install 'decider[serve-sanic]'" in result.output
