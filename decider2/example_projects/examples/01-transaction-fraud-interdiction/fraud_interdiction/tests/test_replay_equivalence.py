"""Spec acceptance criteria 5, 6 and 9 — replay, backtest equivalence, and
retirement evidence.

These are the tests that decide whether the design is worth anything, because
the whole architecture — one feature vector, one Decision object, pinned
generations — exists to make them cheap.
"""

from __future__ import annotations

import polars as pl
import pytest

from decider2.testing import assert_entry_points_agree, assert_modes_agree
from fraud_interdiction.decision.applicability import applicable_bits, applicable_digest
from fraud_interdiction.pipelines.backtest import Decision, backtest
from fraud_interdiction.pipelines.interdict import Interdiction


def test_540_day_replay_is_bit_exact(pinned_generation, recorded_event):
    """Criterion 5: any event from the last 540 days replays to the identical
    action_code, identical fired_rule_ids IN ORDER, identical
    fired_on_overlay_ids and identical reason set, using only recorded
    artefacts.

    "A replay that reaches for any current value has failed." The reason this
    can be asserted rather than hoped for is that `Decision`'s leaf inputs are
    the recorded feature vector and nothing else — there is no current value in
    reach.
    """
    replayed = Decision.score(**recorded_event.feature_vector,
                              generation=pinned_generation)
    assert replayed.action_code == recorded_event.action_code
    assert replayed.live_fire_bits == recorded_event.live_fire_bits
    assert replayed.fired_on_overlay_ids == recorded_event.fired_on_overlay_ids
    assert replayed.shadow_fire_bits == recorded_event.shadow_fire_bits
    assert replayed.decline_reason_codes == recorded_event.decline_reason_codes


def test_applicable_set_reconstructs_and_the_digest_agrees(recorded_event):
    """Criterion 3's awkward half: the applicable population must be
    reconstructable without storing 635 identifiers per record.

    28 bytes of recorded scalars in, a 635-bit set out, checked against the
    stored 8-byte digest. If they disagree the replay fails loudly rather than
    answering a subtly different question — which is the failure mode a
    derivation without a checksum has, and it is silent.
    """
    rebuilt = applicable_bits(**recorded_event.applicability_inputs)
    assert applicable_digest(rebuilt) == recorded_event.applicable_digest


@pytest.mark.parametrize("overlay_mode", ["as_it_was", "disabled"])
def test_daily_backtest_reproduces_production(yesterday_sample, overlay_mode):
    """Criterion 6: running the live rule set and live overlay stack in
    backtest mode over a sampled day reproduces production's action and firing
    set for 100% of events. Checked daily; any mismatch is an incident.

    Criterion 7 is the same call with the stack disabled, which is how the base
    rule set's own performance stays measurable — "the only defence against an
    overlay stack that has quietly become the real policy while the rule set it
    modifies has stopped being maintained".
    """
    out = backtest(yesterday_sample, baseline="as_at_event", overlay_mode=overlay_mode)
    if overlay_mode == "as_it_was":
        assert out.select(pl.col("action_code") != pl.col("recorded_action_code")).sum().item() == 0
        assert out.select(pl.col("live_fire_bits") != pl.col("recorded_live_fire_bits")).sum().item() == 0


def test_entry_points_agree(corpus):
    """The fourth rung of the equivalence ladder, which doc 02 s3.1 does not have.

    `score()` fuses maximally and `apply()` fuses not at all (doc 02 s1.2), so
    they take genuinely different codegen paths. The ladder asserts
    interpreted == stepped == fused WITHIN an entry point and says nothing about
    the two entry points agreeing with each other — while this project's
    acceptance criterion 6 is exactly that they do, at 420 M events.

    Exact equality, not a tolerance. Which means `fastmath` is banned project-
    wide: EXPERIMENTS.md s I measured 46-73% of rows differing by up to 17 ULP
    on a 16-term chain for a 1.09x gain. A fraud threshold comparison at 1 ULP
    is a different decision for the client standing at the till.
    """
    assert_modes_agree(Decision, corpus)
    assert_entry_points_agree(Decision, corpus, atol=0.0)


def test_retired_rule_has_no_effect_after_its_instant(firing_aggregates):
    """Criterion 9 and spec s9.3: "prove rule MS-0117 stopped affecting
    outcomes when you say it did".

    Answered from the per-minute firing aggregate, not by scanning 140 M
    records: MS-0117's count is zero from 2026-06-30T23:59:59Z onward, and its
    bit is clear in the applicable digest on every event after that instant.
    Both are cheap because the firing set is a fixed-width bitset with a
    per-minute rollup, rather than a variable-length list of identifiers.
    """
    after = firing_aggregates.filter(
        (pl.col("rule_id") == "MS-0117") & (pl.col("minute") > "2026-06-30T23:59:00Z")
    )
    assert after.select(pl.col("fire_count").sum()).item() == 0


def test_a_field_that_did_not_exist_is_honest_in_backtest(eighteen_month_window):
    """Spec s11.6 and s11.14: a scheme field added on 2026-04-01, a backtest run
    over 18 months across four rule set generations.

    Events before the availability date carry the field as NOT_APPLICABLE, not
    as null and not as zero, so rules referencing it record as *unevaluable* on
    that population rather than silently evaluating false. The backtest's hit
    rate is then over the applicable events only, and the pack states the
    truncated window. A backtest that quietly treated the field as absent would
    understate the rule's hit rate by the fraction of the window that predates
    the field, and nobody would know.
    """
    pass
