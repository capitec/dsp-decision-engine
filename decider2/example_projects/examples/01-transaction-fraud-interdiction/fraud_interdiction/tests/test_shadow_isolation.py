"""Spec acceptance criterion 4: "No shadow rule has ever changed an action_code,
reason set, queue routing or client notification. Demonstrated continuously, not
asserted."

Spec Q9 asks for this "by construction, such that a shadow rule influencing an
outcome is impossible rather than caught". Three tests, in increasing order of
how much they prove, and only the first two are worth having.
"""

from __future__ import annotations

import pytest

from fraud_interdiction.decision.resolve import RESOLVER_READS, Resolve
from fraud_interdiction.pipelines.interdict import Interdiction
from fraud_interdiction.ruleset import LiveRules, ShadowRules


def test_action_code_lineage_excludes_shadow():
    """The whole guarantee, in one static query. No data, no execution.

    `lineage` is a property of the graph (doc 04 s3), so this is answerable at
    import time and it is a *proof over the definition*, not a sample over
    behaviour. If someone adds shadow_fire_bits to the resolver's reads, this
    fails in CI before the branch is reviewable.
    """
    sources = Interdiction.lineage("action_code").sources
    assert not any(s.startswith("shadow_") for s in sources)
    for downstream in ("counterfactual_action_code", "decline_reason_codes",
                       "effective_queue", "challenge_selection", "client_wording",
                       "obligations_raised", "downstream_instructions"):
        assert not any(s.startswith("shadow_") for s in
                       Interdiction.lineage(downstream).sources)


def test_shadow_module_is_topologically_after_resolve():
    """Belt and braces, and it catches the case lineage cannot.

    A shadow rule could in principle influence an outcome through a value that
    lineage does not track — a tap column consumed by a downstream system, say.
    Position in the sequence closes that: nothing produced after Resolve can be
    read by Resolve, whatever it is called.
    """
    order = [m.name for m in Interdiction.modules]
    assert order.index("shadow_rules") > order.index("resolve")
    assert order.index("shadow_rules") > order.index("dispatch")
    assert order.index("shadow_rules") > order.index("wording")


def test_shadow_and_live_write_disjoint_names():
    """Two populations, two output namespaces, no overlap. A union by accident
    downstream is then a missing-column error rather than a wrong answer
    (spec s5.11: 'in a way a downstream consumer cannot accidentally union
    with it')."""
    assert set(LiveRules.writes).isdisjoint(set(ShadowRules.writes))
    assert set(ShadowRules.writes).isdisjoint(set(RESOLVER_READS))


@pytest.mark.skip(reason="Kept to record why it is NOT the test that matters.")
def test_empirical_shadow_isolation_over_a_days_traffic():
    """Replay a day of events with the shadow population emptied and assert the
    action set is unchanged.

    This is what a system without static lineage has to do, and it is worse in
    every way: it costs a day of compute, it proves a property of one day's
    traffic rather than of the definition, and a shadow rule that only matters
    on 0.001% of events passes it. It is recorded here because the difference
    between this test and the first one is the entire argument for structure
    being inspectable data.
    """
