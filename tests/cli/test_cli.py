"""`decikit template`, `build` and `serve` over a generated starter project."""
import json
import os
import subprocess
import sys
import types
from datetime import date

import polars as pl
import pytest
from click.testing import CliRunner

from decikit import SteppedEngine
from decikit._tooling.cli import CPU_TARGET_FILE, cli
from decikit._engine.compile import cpu_target
from decikit._engine.compile.njit import _DISPATCHERS, _REASONS
from decikit.serving._handler import construct_handler_from_settings



@pytest.fixture
def project(tmp_path, monkeypatch):
    # The CLI reads settings from the environment and imports the project's package by name,
    # so each test gets a clean environment, settings object and module cache.
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if not k.upper().startswith("DECIKIT_")})
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [m for m in sys.modules if m.split(".")[0] == "proj"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["template", "proj"])
    assert result.exit_code == 0, result.output
    monkeypatch.chdir(tmp_path / "proj")
    return tmp_path / "proj"


def test_template_writes_a_uniquely_named_package(project):
    files = sorted(p.relative_to(project).as_posix() for p in project.rglob("*") if p.is_file())
    assert files == [".env", "README.md", "__init__.py", "configs/0.0.0/params.json", "conftest.py",
                     "inference.py", "pipeline.py", "sample_request.json", "tests/test_pipeline.py"]
    assert 'name="proj"' in (project / "pipeline.py").read_text()
    assert "from decikit.serving import RequestHandler" in (project / "inference.py").read_text()
    assert "DECIKIT_API__PIPELINE=proj.pipeline:build" in (project / ".env").read_text()


def test_template_sanitises_the_name_into_the_package_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["template", "credit-risk"])
    assert result.exit_code == 0, result.output
    package = tmp_path / "credit_risk"
    assert (package / "__init__.py").exists() and (package / "pipeline.py").exists()
    assert "DECIKIT_API__PIPELINE=credit_risk.pipeline:build" in (package / ".env").read_text()


def test_template_refuses_a_name_that_cant_be_a_package(tmp_path):
    result = CliRunner().invoke(cli, ["template", "01-fraud", str(tmp_path / "p")])
    assert result.exit_code != 0 and "start it with a letter" in result.output


def test_the_template_params_document_matches_the_pipeline(project):
    sys.path.insert(0, str(project.parent))
    from proj.pipeline import build

    params = json.loads((project / "configs/0.0.0/params.json").read_text())
    assert build().parameters().defaults() == params == {
        "proj": {"approved": {"limit": 0.4, "month_end_limit": 0.3}}}


def test_the_generated_tests_pass(project):
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                            cwd=project, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_template_refuses_a_non_empty_directory(project):
    result = CliRunner().invoke(cli, ["template", "again", str(project)])
    assert result.exit_code != 0
    assert "not empty" in result.output


def test_build_stages_warms_and_records_the_cpu_target(project):
    # A build is a fresh process in real use. In-process, a dispatcher or a cached compile
    # verdict from another test means nothing compiles here, so numba would write no .nbi.
    _DISPATCHERS.clear()
    _REASONS.clear()
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code == 0, result.output
    assert "built config version 0.0.0" in result.output
    assert tuple(json.loads((project / CPU_TARGET_FILE).read_text())) == cpu_target()
    assert any(p.suffix == ".nbi" for p in (project / "__pycache__").iterdir())


def test_the_built_project_serves_the_generated_handler(project):
    assert CliRunner().invoke(cli, ["build"]).exit_code == 0
    handler = construct_handler_from_settings()
    assert type(handler).__name__ == "Handler" and type(handler).__module__ == "proj.inference"
    handler.stage()
    handler.activate()
    live = handler.module_fn()
    day = date(2026, 1, 5)
    assert live.executable.run({"income": 1000.0, "debt": 500.0, "applied_on": day}, live.params)["approved"] is False
    assert live.executable.run({"income": 1000.0, "debt": None, "applied_on": day}, live.params)["approved"] is True


def test_build_fails_with_a_clear_message_on_an_invalid_params_document(project):
    (project / "configs/0.0.0/params.json").write_text(json.dumps({"proj": {"approved": {"limit": 7.0}}}))
    result = CliRunner().invoke(cli, ["build"])
    assert result.exit_code != 0
    assert "config version latest failed to build" in result.output
    assert "limit" in result.output


def test_build_fails_cleanly_on_an_unknown_version(project):
    result = CliRunner().invoke(cli, ["build", "9.9.9"])
    assert result.exit_code != 0
    assert "9.9.9 not found" in result.output


def test_build_fails_cleanly_when_the_pipeline_is_not_importable(project):
    result = CliRunner().invoke(cli, ["build"], env={"DECIKIT_API__PIPELINE": "nowhere:build"})
    assert result.exit_code != 0
    assert "No module named 'nowhere'" in result.output


def test_the_environment_wins_over_the_project_env_file(project):
    with open(project / "pipeline.py", "a") as f:
        f.write("\n\ndef other():\n    return flow(debt_ratio, name='other')\n")
    (project / "configs/0.0.0/params.json").write_text("{}")
    result = CliRunner().invoke(cli, ["build"], env={"DECIKIT_API__PIPELINE": "proj.pipeline:other"})
    assert result.exit_code == 0, result.output
    assert "pipeline proj.pipeline:other" in result.output


def test_decikit_help_points_to_the_guide():
    result = CliRunner().invoke(cli, ["--help"])
    assert "decikit guide" in result.output and "guide" in result.output.split("Commands:")[1]


@pytest.fixture
def uvicorn_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **kw: calls.append((a, kw))))
    return calls


def test_serve_starts_the_starlette_factory_with_the_given_settings(project, uvicorn_calls):
    result = CliRunner().invoke(cli, ["serve", "--port", "9001", "--workers", "3", "--engine", "stepped"])
    assert result.exit_code == 0, result.output
    assert uvicorn_calls == [(("decikit.serving._servers.starlette:create_app",),
                              {"factory": True, "host": "0.0.0.0", "port": 9001, "workers": 3})]
    # What each worker's app factory builds its handler from; the pipeline comes from the project's .env.
    handler = construct_handler_from_settings()
    assert (handler.engine, handler.pipeline) == (SteppedEngine(), "proj.pipeline:build")
    assert os.environ["DECIKIT_API__ENGINE__TYPE"] == "stepped"


def test_serve_refuses_an_unknown_engine_before_starting_any_worker(project, uvicorn_calls, monkeypatch):
    result = CliRunner().invoke(cli, ["serve", "--engine", "fsued"])
    assert result.exit_code == 1 and "Did you mean 'fused'" in result.output
    monkeypatch.setitem(os.environ, "DECIKIT_API__ENGINE__STRICT", "true")
    result = CliRunner().invoke(cli, ["serve", "--engine", "fused"])
    assert result.exit_code == 1 and "strict" in result.output and "Extra inputs" in result.output
    assert uvicorn_calls == []


def test_serve_and_build_refuse_the_polars_engine_before_any_worker_starts(project, uvicorn_calls):
    result = CliRunner().invoke(cli, ["serve", "--engine", "polars"])
    assert result.exit_code == 1 and "numba engine" in result.output and uvicorn_calls == []
    result = CliRunner().invoke(cli, ["build"], env={"DECIKIT_API__ENGINE__TYPE": "polars"})
    assert result.exit_code == 1 and "numba engine" in result.output


def test_serve_warns_when_the_build_ran_on_another_cpu(project, uvicorn_calls):
    triple, _, features = cpu_target()
    (project / CPU_TARGET_FILE).write_text(json.dumps([triple, "some-other-cpu", features]))
    result = CliRunner().invoke(cli, ["serve", "--workers", "1"])
    assert result.exit_code == 0, result.output
    assert "some-other-cpu" in result.output and "decikit build" in result.output
    assert len(uvicorn_calls) == 1


def test_serve_is_quiet_on_the_cpu_it_was_built_for(project, uvicorn_calls):
    assert CliRunner().invoke(cli, ["build"]).exit_code == 0
    result = CliRunner().invoke(cli, ["serve", "--workers", "1"])
    assert result.exit_code == 0 and "warning" not in result.output


def test_serve_with_sanic_missing_is_a_clean_error(project, monkeypatch):
    monkeypatch.setitem(sys.modules, "sanic", None)
    result = CliRunner().invoke(cli, ["serve", "--server", "sanic"])
    assert result.exit_code != 0
    assert "pip install 'decikit[serve-sanic]'" in result.output


def _experiment_json(tmp_path, monkeypatch, limit_ok=True):
    (tmp_path / "xp_credit.py").write_text(
        "from decikit import flow\n"
        "def ratio(income, debt):\n"
        "    return debt / income\n"
        "pipeline = flow(ratio, name='credit')\n"
    )
    pl.DataFrame({"income": [1000.0, 1000.0 if limit_ok else 0.0], "debt": [200.0, 350.0]}).write_parquet(
        tmp_path / "loans.parquet")
    file = tmp_path / "experiment.json"
    file.write_text(json.dumps({"name": "xp", "flow": {"entry": "xp_credit:pipeline"}, "input": {"path": "loans.parquet"}}))
    monkeypatch.syspath_prepend(str(tmp_path))
    return file


def test_experiment_run_exits_zero_when_completed_and_writes_results(tmp_path, monkeypatch):
    file = _experiment_json(tmp_path, monkeypatch)
    result = CliRunner().invoke(cli, ["experiment", "run", str(file), "--out", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "xp: completed" in result.output
    assert (tmp_path / "out" / "result.json").exists()


def test_experiment_run_exits_one_when_the_run_is_not_completed(tmp_path, monkeypatch):
    file = _experiment_json(tmp_path, monkeypatch, limit_ok=False)  # divides by zero: the baseline fails
    result = CliRunner().invoke(cli, ["experiment", "run", str(file)])

    assert result.exit_code == 1
    assert "xp: failed" in result.output
