"""Aggregate summaries over an experiment's scenarios, computed at run time.

The run keeps large populations aggregated: `summarise` reduces each scenario's
output to per-column changed/unchanged/unique counts, the first divergence, and
a sampled drill-down of changed values, never materialising per-record graph
state. `path_counts` records how many records reached each branch/tree position.
"""
from __future__ import annotations

from typing import Any

from decider.experiments.compare import diff_outputs
from decider.experiments.model import ComparisonPolicy

DRILL_DOWN_SAMPLES = 5


def summarise(def_: Any, baseline: dict, scenarios: list[dict]) -> dict[str, Any]:
    """Reduce `baseline` and each scenario's trace to aggregate summary data.

    Args:
        def_: the `ExperimentDef`, for the comparison policy and outputs.
        baseline: the baseline trace (`{"output", "steps", "paths", "error"}`).
        scenarios: one trace per scenario, with `"name"` and `"status"` added.

    Returns a JSON-ready dict: per-scenario changed/unchanged/unique values,
    first divergence, path counts, selected-step differences and a sampled
    drill-down of changed rows.
    """
    comparison: ComparisonPolicy = def_.comparison
    outputs = list(comparison.outputs or tuple(baseline.get("output") or ()))
    rows = [_scenario_summary(sc, baseline, comparison, outputs) for sc in scenarios]
    first = _first_divergence(rows)
    return {
        "outputs": outputs,
        "first_divergence": first,
        "scenarios": rows,
    }


def _scenario_summary(sc: dict, baseline: dict, comparison: ComparisonPolicy,
                      outputs: list[str]) -> dict[str, Any]:
    changed = diff_outputs(baseline.get("output"), sc.get("output"), comparison)
    baseline_output = baseline.get("output") or {}
    scenario_output = sc.get("output") or {}
    unchanged = {
        col: (len(baseline_output[col]) - len(changed.get(col, ())))
        for col in outputs if col in baseline_output
    }
    unique = {
        col: len(set(_scalar(v) for v in scenario_output[col]))
        for col in outputs if col in scenario_output
    }
    drill = {
        col: [{"row": i, "expected": baseline_output[col][i], "actual": scenario_output[col][i]}
              for i in changed[col][:DRILL_DOWN_SAMPLES]]
        for col in changed if col in scenario_output
    }
    return {
        "name": sc["name"],
        "status": sc.get("status"),
        "error": sc.get("error"),
        "changed": changed,
        "unchanged": unchanged,
        "unique": unique,
        "path_counts": sc.get("paths", {}),
        "step_differences": _step_differences(baseline.get("steps", {}), sc.get("steps", {})),
        "drill_down": drill,
    }


def _step_differences(base_steps: dict, sc_steps: dict) -> list[str]:
    out = []
    for path in sorted(set(base_steps) | set(sc_steps)):
        if base_steps.get(path) != sc_steps.get(path):
            out.append(path)
    return out


def _first_divergence(scenarios: list[dict]) -> dict[str, str] | None:
    for summary in scenarios:
        for col, rows in summary["changed"].items():
            if rows:
                return {"scenario": summary["name"], "location": f"{col}[{rows[0]}]"}
    return None


def _scalar(v: Any) -> Any:
    return v if not isinstance(v, float) else round(v, 12)
