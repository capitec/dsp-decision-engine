from __future__ import annotations

import json
import sys

import polars as pl

from decider import branch, flow, param, step
from decider.experiments import (Experiment, ExperimentDef, RunStatus, ScenarioStatus, dump_yaml, load_yaml, resolve,
                                 run, run_experiment)


def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


def _pipeline():
    return flow(ratio, approved, tier, name="demo")


def _frame():
    return pl.DataFrame({"income": [1000.0] * 4, "debt": [float(i * 200) for i in range(4)]})


def _experiment(**kwargs):
    return (Experiment("demo-drift", _pipeline(), _frame(), **kwargs)
            .scenario("baseline")
            .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
            .compare(outputs=("tier",)))


def test_summary_reports_changed_unchanged_unique_and_first_divergence():
    result = _experiment().run()

    assert result.status is RunStatus.COMPLETED
    summary = result.summary
    assert summary["outputs"] == ["tier"]
    assert summary["first_divergence"] == {"scenario": "limit_0.3", "location": "tier[2]"}
    baseline, variant = summary["scenarios"]
    assert baseline["changed"] == {}
    assert baseline["unchanged"]["tier"] == 4
    assert variant["changed"]["tier"] == [2]
    assert variant["unchanged"]["tier"] == 3
    assert variant["unique"]["tier"] == 2
    assert variant["drill_down"]["tier"][0]["expected"] == 1.0
    assert variant["drill_down"]["tier"][0]["actual"] == 0.5


def test_failed_scenario_leaves_the_run_partial_not_completed():
    exp = (Experiment("demo-drift", _pipeline(), _frame())
           .scenario("baseline")
           .scenario("bad_cast", at="demo/ratio", overrides={"ratio": "not-a-float"})
           .compare(outputs=("tier",)))
    result = exp.run()

    assert result.status is RunStatus.PARTIAL
    assert result.scenarios[1].status is ScenarioStatus.FAILED
    assert result.summary["scenarios"][1]["error"]


def test_sankey_counts_every_record_once_through_branches_and_outcomes():
    def is_private(sector_code: int) -> bool:
        return sector_code == 1

    @step(output="term_cap")
    def cap_a(term_cap: float, cap: float = param(54.0)) -> float:
        return min(term_cap, cap)

    @step(output="term_cap")
    def cap_b(term_cap: float, cap: float = param(60.0)) -> float:
        return min(term_cap, cap)

    pipeline = flow(branch(is_private, cap_a, cap_b, modifies=["term_cap"], name="by_sector"), name="demo")
    frame = pl.DataFrame({"sector_code": [1, 2], "term_cap": [72.0, 72.0]})
    exp = (Experiment("branch", pipeline, frame)
           .scenario("baseline")
           .compare(outputs=("term_cap",)))
    result = exp.run()

    assert result.sankey is not None
    assert result.sankey["outcome"] == "term_cap"
    assert sum(link["value"] for link in result.sankey["links"]) == 2 * frame.height
    arm_nodes = [n for n in result.sankey["nodes"] if n["kind"] == "arm"]
    assert len(arm_nodes) == 2  # private and public arms


def test_save_writes_outputs_manifest_summary_and_sankey(tmp_path):
    exp = _experiment()
    result = run_experiment(exp.def_, exp.step, exp.frame, out_dir=str(tmp_path))

    assert (tmp_path / "baseline.parquet").exists()
    assert (tmp_path / "limit_0.3.parquet").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["kind"] == "experiment_run"
    assert json.loads((tmp_path / "summary.json").read_text())["outputs"] == ["tier"]
    assert json.loads((tmp_path / "sankey.json").read_text())["outcome"] == "tier"
    assert result.scenarios[0].output_ref.location.endswith("baseline.parquet")


def test_plain_python_caller_runs_an_experiment_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "credit.py").write_text(
        "from decider import flow, param\n"
        "def ratio(income, debt):\n"
        "    return debt / income if income > 0 else 1.0\n"
        "def approved(ratio, limit=param(0.4)):\n"
        "    return ratio <= limit\n"
        "pipeline = flow(ratio, approved, name='credit')\n"
    )
    (tmp_path / "loans.json").write_text(
        '[{"income": 1000.0, "debt": 200.0}, {"income": 1000.0, "debt": 350.0}]'
    )
    yaml_text = dump_yaml(ExperimentDef(
        name="drift",
        flow={"entry": "credit:pipeline"},
        input={"path": str(tmp_path / "loans.json"), "id_column": None},
        scenarios=[{"name": "baseline"}, {"name": "limit_0.3", "params": {"credit": {"approved": {"limit": 0.3}}}}],
        comparison={"outputs": ("approved",)},
    ).model_dump())
    monkeypatch.syspath_prepend(str(tmp_path))

    result = run(ExperimentDef.model_validate(load_yaml(yaml_text)))

    assert result.status is RunStatus.COMPLETED
    assert result.manifest.input.dataset == "loans.json"
    assert [s.name for s in result.scenarios] == ["baseline", "limit_0.3"]
    assert result.scenarios[1].divergences


def test_resolve_revision_turns_head_symbols_into_a_sha():
    sha = resolve("HEAD^")
    assert sha and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)
    assert resolve("HEAD") == resolve("HEAD")
