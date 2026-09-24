"""Integration: the one servable capability (`pipeline.py`), end to end, exactly the
way `decider build`/`decider serve` would run it."""
import json
import os
import subprocess
from pathlib import Path

import pytest
from decider import Engine

import capture_demo_evidence as cde
import pipeline

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _seeded_evidence():
    """The servable pipeline replays a decision by id -- it needs that decision's
    evidence captured first, exactly as SERVE.md's `capture_demo_evidence.py` step
    describes for a fresh checkout."""
    cde.main()


def test_the_sample_request_reproduces_through_the_bound_pipeline():
    exe = Engine().bind(pipeline.build(), mode="interpreted")
    record = json.loads((ROOT / "sample_request.json").read_text())
    out = exe.score(record, pipeline.build().parameters().defaults())
    assert out["replay_verdict"] == "reproduced"
    assert out["divergence_count"] == 0
    assert out["diverged_fields"] == ""


def test_a_decision_that_does_not_exist_raises_rather_than_fabricating_a_verdict():
    exe = Engine().bind(pipeline.build(), mode="interpreted")
    with pytest.raises(FileNotFoundError):
        exe.score({"flow_code": "03", "decision_id": "does-not-exist"}, {})


def test_decider_build_cli_succeeds():
    """Verifies serving exactly as SERVE.md documents: the real CLI, not a Python shortcut."""
    import decider as _decider

    from conftest import _find_sibling, _SIBLINGS

    repo_root = Path(_decider.__file__).resolve().parents[1]  # .../<repo>/decider/__init__.py

    env = dict(os.environ)
    env["DECIDER_API__CODE_PATH"] = str(ROOT)
    env["DECIDER_API__PIPELINE"] = "pipeline:build"
    env["DECIDER_CONFIG__BASEPATH"] = str(ROOT / "configs")
    env["DECIDER_API__MODE"] = "interpreted"
    env["PYTHONPATH"] = os.pathsep.join(str(_find_sibling(name)) for name in _SIBLINGS)

    # `decider build`'s warm-up scores the sample request (`inference.py`), which needs
    # that decision's evidence on disk -- SERVE.md's own "populate the evidence store
    # first" step, run here for real (not through the `_isolated_evidence_store` fixture,
    # which only patches this test *process*, not the subprocess below).
    capture = subprocess.run(["uv", "run", "--project", str(repo_root), "python", "capture_demo_evidence.py"],
                              cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)
    assert capture.returncode == 0, capture.stdout + capture.stderr

    result = subprocess.run(["uv", "run", "--project", str(repo_root), "decider", "build"],
                             cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "built config version" in (result.stdout + result.stderr)
