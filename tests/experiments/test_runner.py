from __future__ import annotations

import subprocess

import polars as pl

from decider import flow
from decider.experiments import (Callback, Divergence, Experiment, ExperimentDef, RunStatus, ScenarioStatus,
                                 ValueCounts, resolve, run)
from decider.lifecycle import JobHandle

LIMIT_03 = {"demo": {"approved": {"limit": 0.3}}}


class Record(Callback):
    def __init__(self):
        self.batches, self.done = [], []

    def on_batch_end(self, batch):
        self.batches.append((batch.scenario, batch.index, batch.offset, batch.output.height))

    def on_progress(self, run, done):
        self.done.append(done)


def test_baseline_and_a_variant_run_with_a_manifest(pipeline, frame):
    result = (Experiment("demo-drift", pipeline, frame)
              .scenario("limit_0.3", params=LIMIT_03)
              .metric(Divergence(column="tier"))
              .run())

    assert result.status is RunStatus.COMPLETED
    assert not result.nondeterministic
    assert [(s.name, s.status, s.records) for s in result.scenarios] == [
        ("baseline", ScenarioStatus.COMPLETED, 4), ("limit_0.3", ScenarioStatus.COMPLETED, 4)]
    assert result.summaries["divergence"]["limit_0.3"].changed == 1
    assert result.manifest.kind == "experiment_run"
    assert result.manifest.input.row_count == 4
    assert result.manifest.revision.authored == "HEAD"
    assert result.preflight.ok


def test_every_scenario_runs_each_batch_before_the_next_batch_starts(pipeline, frame):
    record = Record()
    Experiment("x", pipeline, frame, batch_size=3).scenario("limit_0.3", params=LIMIT_03).callback(record).run()

    assert record.batches == [("baseline", 0, 0, 3), ("limit_0.3", 0, 0, 3),
                              ("baseline", 1, 3, 1), ("limit_0.3", 1, 3, 1)]
    assert record.done == [3, 4]


def test_a_lazy_frame_gives_the_same_summaries_as_an_eager_one(pipeline, frame):
    def summaries(f):
        exp = Experiment("x", pipeline, f, batch_size=3).scenario("limit_0.3", params=LIMIT_03)
        return exp.metric(Divergence(column="tier")).run().summaries

    assert summaries(frame.lazy()) == summaries(frame)


def test_a_failed_scenario_leaves_the_run_partial_not_completed(pipeline, frame):
    result = (Experiment("x", pipeline, frame)
              .scenario("bad_cast", at="demo/ratio", overrides={"ratio": "not-a-float"})
              .metric(Divergence(column="tier"))
              .run())

    assert result.status is RunStatus.PARTIAL
    assert result.scenarios[1].status is ScenarioStatus.FAILED
    assert "doesn't cast" in result.scenarios[1].error
    assert list(result.summaries["divergence"]) == ["baseline"]


def test_a_failing_baseline_fails_the_run(frame):
    def broken(income: float) -> float:
        raise ZeroDivisionError("boom")

    result = Experiment("x", flow(broken, name="demo"), frame).run()
    assert result.status is RunStatus.FAILED
    assert "boom" in result.scenarios[0].error


def test_a_nondeterministic_flow_is_non_reproducible(frame):
    counter = {"n": 0}

    def drift(income: float, debt: float) -> float:
        counter["n"] += 1
        return debt / income + counter["n"] * 1e-12

    result = Experiment("impure", flow(drift, name="demo"), frame).run()
    assert result.status is RunStatus.NON_REPRODUCIBLE
    assert result.nondeterministic


def test_an_override_at_a_step_is_what_later_steps_read(pipeline, frame):
    result = (Experiment("x", pipeline, frame)
              .scenario("force_ratio", at="demo/ratio", overrides={"ratio": 0.1})
              .metric(ValueCounts(column="approved", nodes="demo/approved"))
              .run())
    assert result.summaries["value_counts"]["force_ratio"].counts == {"True": 4}


def test_a_row_override_changes_only_that_record_even_in_a_later_batch(pipeline, frame):
    result = (Experiment("x", pipeline, frame, batch_size=2)
              .scenario("poor", overrides={"income": 100.0}, row=1)
              .metric(Divergence(column="tier"))
              .run())
    poor = result.summaries["divergence"]["poor"]
    assert (poor.changed, poor.first) == (1, 1)


def test_a_job_reports_progress_and_cancelling_it_stops_after_the_batch(pipeline, frame):
    class CancelAfterFirst(Callback):
        def on_batch_end(self, batch):
            job.cancel()

    job = JobHandle(job_id="j1", kind="experiment_run")
    record = Record()
    result = Experiment("x", pipeline, frame, batch_size=1).callback(CancelAfterFirst(), record).run(job=job)

    assert result.status is RunStatus.CANCELLED
    assert record.done == [1]


def test_out_dir_gets_each_scenarios_output_batches_and_the_result(pipeline, frame, tmp_path):
    result = (Experiment("x", pipeline, frame, batch_size=3)
              .scenario("limit_0.3", params=LIMIT_03)
              .run(out_dir=str(tmp_path)))

    assert pl.scan_parquet(tmp_path / "limit_0.3" / "*.parquet").collect().height == 4
    assert result.scenarios[1].output_ref.location == str(tmp_path / "limit_0.3")
    saved = (tmp_path / "result.json").read_text()
    assert result.model_validate_json(saved).status is RunStatus.COMPLETED


def test_a_plain_python_caller_runs_a_json_definition_end_to_end(tmp_path, monkeypatch):
    (tmp_path / "credit.py").write_text(
        "from decider import flow, param\n"
        "def ratio(income, debt):\n"
        "    return debt / income if income > 0 else 1.0\n"
        "def approved(ratio, limit=param(0.4)):\n"
        "    return ratio <= limit\n"
        "pipeline = flow(ratio, approved, name='credit')\n"
    )
    pl.DataFrame({"income": [1000.0, 1000.0], "debt": [200.0, 350.0]}).write_parquet(tmp_path / "loans.parquet")
    (tmp_path / "experiment.json").write_text(ExperimentDef(
        name="drift",
        flow={"entry": "credit:pipeline"},
        input={"path": str(tmp_path / "loans.parquet")},
        scenarios=[{"name": "limit_0.3", "params": {"credit": {"approved": {"limit": 0.3}}}}],
        metrics=[{"type": "divergence", "column": "approved"}],
    ).model_dump_json())
    monkeypatch.syspath_prepend(str(tmp_path))

    result = run(ExperimentDef.model_validate_json((tmp_path / "experiment.json").read_text()))

    assert result.status is RunStatus.COMPLETED
    assert result.manifest.input.dataset == "loans.parquet"
    assert result.summaries["divergence"]["limit_0.3"].changed == 1


def test_resolve_turns_head_symbols_into_a_sha():
    sha = resolve("HEAD^")
    assert sha and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_a_revision_scenario_runs_the_checked_out_code_even_with_the_live_module_imported():
    # Building the experiment imports the live pipeline under the same name the revision's
    # checkout would load it by. The file's last commit added a flat $50 fee to `ratio`.
    import example_projects.experiments.credit_pipeline as live

    fee = subprocess.run(["git", "log", "-1", "--format=%H", "--", live.__file__],
                         capture_output=True, text=True, check=True).stdout.strip()

    frame = pl.DataFrame({"income": [1000.0, 2000.0, 1500.0, 800.0], "debt": [350.0, 900.0, 700.0, 300.0]})
    entry = "example_projects.experiments.credit_pipeline:build"
    result = (Experiment("version-drift", live.build(), frame, entry=entry, batch_size=3)
              .scenario("previous", revision=f"{fee}^")
              .metric(Divergence(column="approved"), Divergence(name="ratio", column="ratio", nodes="credit/ratio"))
              .run())

    assert result.status is RunStatus.COMPLETED, result.scenarios
    assert result.scenarios[1].revision.resolved == resolve(f"{fee}^")
    assert result.summaries["divergence"]["previous"].changed == 1  # the fee pushes one record over the limit
    assert result.summaries["ratio"]["previous"].changed == 4       # and changes every ratio
