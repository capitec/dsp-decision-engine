"""End-to-end: `decider build` (the CLI, as SERVE.md documents it) and
scoring `sample_request.json` through the staged handler -- mirrors
`tests/serving/` and `tests/cli/` in the main repo, and 00/02's own
`test_pipeline.py`."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from decider import Engine

PROJECT_DIR = Path(__file__).resolve().parents[1]
# SERVE.md's own PYTHONPATH (00 and 02, read-only) is what every command in this test
# should use. If the caller already set it (SERVE.md's documented invocation, or this
# repo's `example_projects/<NN>/sonnet` layout), honour it; otherwise fall back to the
# sibling layout this project was developed against.
_DEFAULT_DEPS = os.pathsep.join([
    str(PROJECT_DIR.parents[1] / "00-shared-credit-core" / "sonnet"),
    str(PROJECT_DIR.parents[1] / "02-affordability" / "sonnet"),
])


def _env():
    env = dict(os.environ)
    env["DECIDER_API__CODE_PATH"] = str(PROJECT_DIR)
    env["DECIDER_API__PIPELINE"] = "pipeline:build"
    env["DECIDER_CONFIG__BASEPATH"] = str(PROJECT_DIR / "configs")
    env["DECIDER_API__MODE"] = "fused"
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH") or _DEFAULT_DEPS
    return env


def test_decider_build_succeeds():
    result = subprocess.run(
        [sys.executable, "-c", "from decider.cli import cli; cli()", "build"],
        cwd=PROJECT_DIR, env=_env(), capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "built config version" in (result.stdout + result.stderr)


def test_sample_request_scores_through_the_handler(monkeypatch):
    # PYTHONPATH (00 and 02) is assumed already set process-wide, exactly as SERVE.md's
    # `pytest` invocation sets it -- see this module's `_env()`/`_DEFAULT_DEPS`.
    for k, v in _env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(PROJECT_DIR))
    monkeypatch.chdir(PROJECT_DIR)
    from decider.serving.handler import construct_handler_from_settings
    handler = construct_handler_from_settings()
    handler.stage()
    handler.activate()
    live = handler.module_fn()

    import inference
    record = inference._typed_sample_record()
    result = live.executable.score(record, live.params)

    assert result["matrix_cell_id"].startswith("limit_matrix@")
    assert result["binding_cap_code"] in (0, 1, 2, 3, 5, 6, 7)
    assert result["final_proposed_limit"] >= result["current_limit"]
    assert isinstance(result["exclusion_codes"], list)
    assert result["matrix_multiplier_unadjusted"] >= 1.0


def test_replay_reproduces_the_decision_to_the_cent(monkeypatch, sample_record):
    """09 §5.15 item 6: same snapshot, same artefact versions -> identical output."""
    for k, v in _env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(PROJECT_DIR))
    monkeypatch.chdir(PROJECT_DIR)
    from decider.serving.handler import construct_handler_from_settings
    handler = construct_handler_from_settings()
    handler.stage(); handler.activate()
    live = handler.module_fn()

    first = live.executable.score(sample_record, live.params)
    second = live.executable.score(sample_record, live.params)
    assert first["final_proposed_limit"] == second["final_proposed_limit"]
    assert first["matrix_cell_id"] == second["matrix_cell_id"]
    assert first["binding_cap_code"] == second["binding_cap_code"]
    assert first["probability_of_default"] == second["probability_of_default"]


def test_stack_disabled_is_the_same_apply_stack_call(monkeypatch):
    """§10 item 5: "the cycle can be run with the overlay stack disabled... without a
    separate implementation." Every overlay in this project (`limit_mgmt/overlays.py`)
    is `AdjustmentRegister.apply_stack` with `stack_enabled` flipped -- proved directly
    on the register 07 owns (the matrix dial), the same call the pipeline's
    `apply_stack_step` makes under the hood."""
    monkeypatch.syspath_prepend(str(PROJECT_DIR))
    from limit_mgmt.overlays import MATRIX_ADJUSTMENTS, ADJUSTMENT_SET_ID
    import datetime
    in_force_date = datetime.date(2026, 9, 24)
    on = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_multiplier_excess", 0.50, {}, in_force_date, ADJUSTMENT_SET_ID, stack_enabled=True,
    )
    off = MATRIX_ADJUSTMENTS.apply_stack(
        "matrix_multiplier_excess", 0.50, {}, in_force_date, ADJUSTMENT_SET_ID, stack_enabled=False,
    )
    assert on.adjusted_value == pytest.approx(0.40)   # 80% dial on the excess
    assert off.adjusted_value == pytest.approx(0.50)  # unadjusted, same call, flag flipped
    assert on.unadjusted_value == off.unadjusted_value == pytest.approx(0.50)
