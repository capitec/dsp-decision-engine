"""`Pipeline.serve()` / `ServeHandle` (doc 03 §6, doc 08 §4/§4.1/§4.1b) —
the library-level surface `decider2/serving/` is built on. `tests/
test_serving.py` covers the HTTP layer; this file covers the handle
directly, including the one property the task calls out explicitly: a
params-only activation must never recompile.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from decider2.compile.driver import build_driver
from decider2.examples.flagship import pipeline as flagship_pipeline
from decider2.runtime.invoke import DEFAULT_BUILD_DIR
from decider2.runtime.serve import ChangeClass, SealedModeError, ServeHandle, structure_fingerprint

RECORD = {
    "net_income": 4100.0,
    "expenses": 1500.0,
    "instalment": 800.0,
    "term_cap": 60.0,
    "min_net_salary": 4100.0,
}


def _driver_for(pipeline):
    steps, group_ids, owners, _ = pipeline.flatten_for_runtime()
    terminal_names = frozenset(pipeline.interface.terminals)
    return build_driver(
        list(steps), list(group_ids), owners=list(owners),
        build_dir=DEFAULT_BUILD_DIR, terminal_names=terminal_names,
    )


def test_pipeline_serve_returns_a_serve_handle():
    handle = flagship_pipeline.serve()
    assert isinstance(handle, ServeHandle)
    assert handle.mode == "sealed"


def test_serve_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        flagship_pipeline.serve(mode="fast")


def test_params_only_activation_does_not_recompile():
    """The task's own acceptance bar: assert this by checking the driver's
    signature count is unchanged across an activate."""
    handle = flagship_pipeline.serve()
    handle.score(RECORD)  # warm the path exactly as /invocations would

    driver_before = _driver_for(flagship_pipeline)
    before = len(driver_before.signatures)
    assert before >= 1

    plan = handle.stage({"cap_by_income_band": {"cap": 24.0}})
    assert plan.klass is ChangeClass.VALUES
    assert plan.recompiles is False
    handle.activate()

    driver_after = _driver_for(flagship_pipeline)
    after = len(driver_after.signatures)

    assert after == before
    # Not just equal in count — the literal same cached artefact (doc 08
    # §4's "no compile" claim), because compile.driver.build_driver's cache
    # key never includes a params value.
    assert driver_after is driver_before

    assert handle.score(RECORD)["cap_by_income_band"] == 24.0


def test_stage_validates_before_returning_a_plan():
    handle = flagship_pipeline.serve()
    with pytest.raises(ValidationError):
        handle.stage({"cap_by_income_band": {"cap": 999.0}})
    # a rejected stage must not become pending.
    assert handle.pending is None


def test_stage_rejects_an_unknown_module():
    handle = flagship_pipeline.serve()
    with pytest.raises(ValueError):
        handle.stage({"not_a_module": {"x": 1}})


def test_activate_requires_a_prior_stage():
    handle = flagship_pipeline.serve()
    with pytest.raises(RuntimeError):
        handle.activate()


def test_rollback_requires_a_prior_activate():
    handle = flagship_pipeline.serve()
    with pytest.raises(RuntimeError):
        handle.rollback()


def test_resolved_params_round_trips_through_stage_and_activate():
    handle = flagship_pipeline.serve()
    resolved = handle.resolved_params()
    handle.stage(resolved)
    handle.activate()
    assert handle.resolved_params() == resolved


def test_unmentioned_modules_survive_a_partial_stage():
    """Doc 03 §4.1: a param change's blast radius is one module. Staging a
    change to one module must not disturb another module's own override —
    this fixture only has one tunable module, so the property under test
    is that repeated partial stages to it compose (module-key granularity),
    not that a second module is untouched (see docstring in
    runtime/serve.py `ServeHandle.stage`)."""
    handle = flagship_pipeline.serve()
    handle.stage({"cap_by_income_band": {"cap": 30.0}})
    handle.activate()
    assert handle.resolved_params()["cap_by_income_band"]["cap"] == 30.0

    # Restaging without mentioning income_threshold falls back to ITS OWN
    # model default, not the previously-active value — matching
    # runtime.invoke.resolve_params's own per-call semantics exactly.
    handle.stage({"cap_by_income_band": {"income_threshold": 4000.0}})
    handle.activate()
    resolved = handle.resolved_params()["cap_by_income_band"]
    assert resolved["income_threshold"] == 4000.0
    assert resolved["cap"] == 48.0  # back to the model default, not 30.0


def test_preview_does_not_mutate_state():
    handle = flagship_pipeline.serve()
    before = handle.resolved_params()
    result = handle.preview(RECORD, {"cap_by_income_band": {"cap": 6.0}})
    assert result["current"]["cap_by_income_band"] == 48.0
    assert result["proposed"]["cap_by_income_band"] == 6.0
    assert handle.resolved_params() == before
    assert handle.pending is None


def test_gil_report_flags_every_flagship_kernel_as_holding_the_gil():
    """doc 00 §2c: `nogil` is off by default and authored per step; the
    flagship example declares none, so serving must report every one of
    its kernels as holding the GIL rather than silently assuming otherwise."""
    handle = flagship_pipeline.serve()
    report = handle.gil_report()
    assert report
    assert all(entry["holds_gil"] is True for entry in report)
    assert {"disposable_income", "affordability_ratio", "cap_by_income_band"} <= {
        step for entry in report for step in entry["steps"]
    }


def test_structure_fingerprint_is_stable_and_ignores_param_values():
    a = structure_fingerprint(flagship_pipeline)
    b = structure_fingerprint(flagship_pipeline)
    assert a == b
    handle = flagship_pipeline.serve()
    handle.stage({"cap_by_income_band": {"cap": 6.0}})
    handle.activate()
    assert structure_fingerprint(flagship_pipeline) == a


def test_sealed_mode_refuses_a_plan_that_would_recompile():
    """Unreachable through a real `.stage(doc)` today — decider2 has no
    document-driven way to change structure yet (see runtime/serve.py's
    module docstring) — so this exercises the guard directly rather than
    trying to fabricate a recompiling params document."""
    from decider2.runtime.serve import StagePlan

    handle = flagship_pipeline.serve(mode="sealed")
    plan = StagePlan(klass=ChangeClass.SKELETON, recompiles=True, fingerprint="x", eta=2.5, doc={})
    with pytest.raises(SealedModeError):
        handle._enforce_sealed(plan)

    live_handle = flagship_pipeline.serve(mode="live")
    live_handle._enforce_sealed(plan)  # does not raise
