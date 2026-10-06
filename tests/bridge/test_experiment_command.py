from __future__ import annotations

import io
import json
import threading
import time

import polars as pl

from decider._tooling.bridge.bridge import Bridge


def _definition(tmp_path, monkeypatch, *, rows=2, batch_size=None, **extra):
    (tmp_path / "credit.py").write_text(
        "from decider import flow, param\n"
        "def ratio(income, debt):\n"
        "    return debt / income if income > 0 else 1.0\n"
        "def approved(ratio, limit=param(0.4)):\n"
        "    return ratio <= limit\n"
        "pipeline = flow(ratio, approved, name='credit')\n"
    )
    pl.DataFrame({"income": [1000.0] * rows, "debt": [float(100 + i) for i in range(rows)]}
                ).write_parquet(tmp_path / "loans.parquet")
    doc = {
        "name": "drift",
        "flow": {"entry": "credit:pipeline"},
        "input": {"path": str(tmp_path / "loans.parquet")},
        "scenarios": [{"name": "limit_0.3", "params": {"credit": {"approved": {"limit": 0.3}}}}],
        "metrics": [{"type": "divergence", "column": "approved"}],
        **({"batch_size": batch_size} if batch_size else {}),
        **extra,
    }
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(doc))
    monkeypatch.syspath_prepend(str(tmp_path))
    return path


def test_experiment_schema_lists_registered_types_with_their_summary():
    schema = Bridge().handle({"cmd": "experiment_schema"})
    assert schema["metrics"]["divergence"]["summary"] == "Divergences"
    assert schema["plots"]["changes"]["requires"] == "Divergences"
    assert "overview" in schema["blocks"]


def test_experiment_validate_reports_an_unknown_step_id(tmp_path, monkeypatch):
    path = _definition(tmp_path, monkeypatch,
                       metrics=[{"type": "divergence", "column": "approved", "node_id": "deadbeef0000"}])
    problems = Bridge().handle({"cmd": "experiment_validate", "definition": str(path)})["problems"]
    assert any("deadbeef0000" in p for p in problems)


def test_experiment_validate_passes_a_well_formed_definition(tmp_path, monkeypatch):
    path = _definition(tmp_path, monkeypatch)
    assert Bridge().handle({"cmd": "experiment_validate", "definition": str(path)})["problems"] == []


def test_experiment_runs_and_streams_progress_before_the_final_reply(tmp_path, monkeypatch):
    path = _definition(tmp_path, monkeypatch)
    out = io.StringIO()
    b = Bridge(out=out)
    result = b.handle({"id": 7, "cmd": "experiment", "definition": str(path)})

    assert result["status"] == "completed"
    lines = [json.loads(l) for l in out.getvalue().splitlines()]
    assert lines == [{"id": 7, "progress": {"done": 2, "total": 2}}]


def test_input_overrides_the_definitions_own_dataset(tmp_path, monkeypatch):
    path = _definition(tmp_path, monkeypatch, rows=2)
    other = tmp_path / "other.parquet"
    pl.DataFrame({"income": [1000.0] * 5, "debt": [200.0] * 5}).write_parquet(other)

    result = Bridge().handle({"id": 1, "cmd": "experiment", "definition": str(path), "input": str(other)})

    assert result["manifest"]["input"]["row_count"] == 5


def test_cancel_stops_a_running_experiment_at_the_next_batch(tmp_path, monkeypatch):
    path = _definition(tmp_path, monkeypatch, rows=50, batch_size=1)
    b = Bridge()

    def cancel_once_registered():
        while 7 not in b._jobs:
            time.sleep(0.001)
        b._jobs[7].cancel()

    threading.Thread(target=cancel_once_registered).start()
    result = b.handle({"id": 7, "cmd": "experiment", "definition": str(path)})

    assert result["status"] == "cancelled"
    assert 7 not in b._jobs  # unregistered once the run ends
