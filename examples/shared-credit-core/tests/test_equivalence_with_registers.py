"""The data steps answer exactly as the register classes they replace did, interpreted and fused.

Delete this file with `AdjustmentRegister`, `ReasonCodeRegistry` and `EffectiveDatedSet`.
"""
import itertools
from datetime import date

import pytest
from decider import flow

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister, AdjustmentStack
from credit_core.dates import EffectiveDatedSet, EffectiveVersion, EffectiveVersions
from credit_core.deductions import tax_table_version_step
from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry, ReasonCodes

MODES = ("interpreted", "fused")
DATES = [date(2025, 12, 1), date(2026, 1, 1), date(2026, 6, 1), date(2027, 6, 1), date(2027, 7, 1)]


def register_of(stack):
    return AdjustmentRegister([
        Adjustment(a.adjustment_id, a.kind, a.target, AdjustmentEffect(a.effect.op, a.effect.value), a.scope,
                   a.stack_position, a.owner, a.approval_reference, a.rationale, a.effective_from, a.effective_to,
                   a.review_date, a.tighten_only, a.enabled)
        for a in stack.adjustments])


def outcome(pipeline, mode, record, params):
    try:
        return pipeline.compile(mode).score(record, params)
    except Exception as e:  # the two designs must also fail alike
        return type(e).__name__


@pytest.mark.parametrize("mode", MODES)
def test_the_stack_step_matches_the_register_step_on_90_records(mode, score_adjustments):
    new = flow(score_adjustments, name="t")
    old = flow(register_of(score_adjustments).apply_stack_step("score", score_adjustments.set_id, base_field="score_raw"),
               name="t")
    records = [({"score_raw": 612.5, "decision_date": on, "segment_code": segment, "channel_code": channel,
                 "product_code": 10.0, "scorecard_id": 1}, enabled)
               for on, segment, channel, enabled in itertools.product(DATES, [3, 1, None], [4, 1, None], [True, False])]
    assert len(records) == 90
    answers = []
    for record, enabled in records:
        params = {"t": {"apply_score_adjustments": {"adjustment_stack_enabled": enabled}}}
        answers.append(outcome(new, mode, record, params))
        assert answers[-1] == outcome(old, mode, record, params), (record, enabled)
    assert {tuple(a["adjustments_applied"]) for a in answers} == {(), ("ADJ-2026-001",), ("ADJ-2026-002",),
                                                                   ("ADJ-2026-001", "ADJ-2026-002")}


@pytest.mark.parametrize("mode", MODES)
def test_the_reason_codes_step_matches_the_registry_step(mode, reason_codes):
    registry = ReasonCodeRegistry(reason_codes.version, [ReasonCode(c.code, c.severity_rank, c.description, c.is_regulatory)
                                                         for c in reason_codes.codes])
    new, old = flow(reason_codes, name="t"), flow(registry.resolve_step(), name="t")
    known = [c.code for c in reason_codes.codes]
    for fired in [[], [known[-1]], known[::-1], known + known, [known[0], 9999], [9999]]:
        record = {"decline_reason_codes": fired}
        assert outcome(new, mode, record, None) == outcome(old, mode, record, None), fired


@pytest.mark.parametrize("mode", MODES)
def test_the_effective_versions_step_matches_the_dated_set_step(mode):
    new_step = tax_table_version_step

    dated = EffectiveDatedSet("tax_table", [EffectiveVersion(v.version_id, v.effective_from, v.effective_to)
                                            for v in new_step.versions])
    new, old = flow(new_step, name="t"), flow(dated.resolver_step(output="tax_table_version"), name="t")
    for on in [date(2024, 1, 1), date(2025, 3, 1), date(2026, 2, 28), date(2026, 3, 1), date(2031, 1, 1)]:
        record = {"decision_date": on}
        assert outcome(new, mode, record, None) == outcome(old, mode, record, None), on


def test_the_old_row_types_are_accepted_by_the_new_steps():
    adjustment = Adjustment("ADJ-1", "score_shift", "score", AdjustmentEffect("add", -10.0), {"segment_code": 3}, 1, "Owner",
                            "CRC-1", "why", date(2026, 1, 1), None, date(2026, 12, 31), tighten_only=True)
    assert AdjustmentStack(target="score", set_id="AS-1", adjustments=[adjustment]).apply(
        600.0, {"segment_code": 3}, date(2026, 6, 1)).adjusted_value == 590.0
    assert ReasonCodes(version="v1", codes=[ReasonCode(7, 1, "Sanctioned", True)]).rank([7]) == ([7], 7)
    assert EffectiveVersions(family="x", versions=[EffectiveVersion("v1", date(2025, 1, 1))]).resolve(
        date(2026, 1, 1)).version_id == "v1"
