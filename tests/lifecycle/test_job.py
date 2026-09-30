from __future__ import annotations

from decider.lifecycle import Job, JobHandle, JobStatus, PartialResult, TERMINAL_STATUSES


def test_job_starts_queued_and_immutable():
    handle = JobHandle(job_id="j1", kind="check_run")
    snap = handle.snapshot()
    assert snap.status is JobStatus.QUEUED
    assert snap.terminal is False


def test_start_marks_running_and_sets_deadline():
    handle = JobHandle(job_id="j1", kind="data_load", timeout=10.0)
    handle.start()
    assert handle.snapshot().status is JobStatus.RUNNING
    assert handle.snapshot().started_at is not None
    assert not handle.timed_out


def test_progress_reports_done_total_and_fraction():
    handle = JobHandle(job_id="j1", kind="experiment_run")
    handle.start()
    handle.progress(2, total=5, message="running scenarios")
    snap = handle.snapshot()
    assert snap.done == 2 and snap.total == 5
    assert snap.fraction == 0.4
    assert snap.message == "running scenarios"


def test_fraction_is_none_when_total_unknown():
    handle = JobHandle(job_id="j1", kind="trace_export")
    handle.start()
    handle.progress(3)
    assert handle.snapshot().fraction is None


def test_cancel_is_terminal_and_one_way():
    handle = JobHandle(job_id="j1", kind="revision_compare")
    handle.start()
    handle.cancel()
    assert handle.cancelled
    assert handle.snapshot().status is JobStatus.CANCELLED
    handle.succeed()
    assert handle.snapshot().status is JobStatus.CANCELLED


def test_fail_records_error():
    handle = JobHandle(job_id="j1", kind="check_run")
    handle.start()
    handle.fail("boom")
    snap = handle.snapshot()
    assert snap.status is JobStatus.FAILED
    assert snap.error == "boom"
    assert snap.finished_at is not None


def test_partial_result_survives_failure():
    handle = JobHandle(job_id="j1", kind="experiment_run")
    handle.start()
    handle.fail("scenario 4 crashed", partial=PartialResult(description="3 of 5 scenarios", completed=3))
    snap = handle.snapshot()
    assert snap.partial is not None
    assert snap.partial.completed == 3


def test_log_appends_in_order():
    handle = JobHandle(job_id="j1", kind="data_load")
    handle.start()
    handle.log("reading")
    handle.log("parsing")
    assert handle.snapshot().logs == ("reading", "parsing")


def test_terminal_statuses_are_exactly_the_expected_set():
    assert TERMINAL_STATUSES == frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMED_OUT})


def test_job_snapshot_round_trips_through_json():
    handle = JobHandle(job_id="j1", kind="experiment_run")
    handle.start()
    handle.succeed(result={"rows": 100})
    restored = Job.model_validate_json(handle.snapshot().model_dump_json())
    assert restored == handle.snapshot()
    assert restored.status is JobStatus.SUCCEEDED
