"""P05 precedence — the verdict over the firing set. Spec 5.6. 17 decision points.

The precedence itself is DATA (a `decision_table`, doc 08 §3.4 — uniform
predicates over a firing-bit array, so it recompiles free of a staged compile)
changed by Financial Crime under four-eyes approval, without a release. This
file is the interface: what the precedence reads and what it must produce, not
the thresholds themselves.

`fraud_verdict_code` is read by six later phases (P09, P11, P14, P16, P17,
P18) — declared as `consumed_by=` on the P05 envelope precisely so that
collapsing this into a pass/fail gate is a build error, not a design review
that gets missed.
"""

from __future__ import annotations

from decider2 import decision_table, module, param

PrecedenceTable = decision_table(
    name="fraud_precedence",
    reads=["f1_firing_bits", "f1_max_severity", "f2_firing_bits", "f2_max_severity",
           "f3_firing_bits", "f3_max_severity", "f4_firing_bits", "f4_max_severity",
           "f5_firing_bits", "f5_max_severity"],
    writes=["weighted_score", "fraud_verdict_code"],
    # Rows: "any F5 hit of severity 4 declines", "two or more F1 firings of
    # severity 3 refer", "weighted score above 0.71 refers, above 0.88
    # declines" — all as table rows, not as a Python precedence chain.
)

def bypass_eligible(requested_amount: float, internal_tenure_months: int,
                    adverse_internal_history: bool,
                    consortium_available: bool = param(True)) -> bool:
    """Under R8 000, >= 30 months' tenure, no adverse history: skip the
    consortium call entirely. 23% of entry point 1. Not a fraud verdict —
    a decision about whether to pay for a call at all."""
    pass  # requested_amount < 8_000_00 and internal_tenure_months >= 30 and not adverse

def verdict_and_reasons(bypass_eligible: bool, weighted_score: float,
                        consortium_available: bool) -> dict:
    """Composes the table's verdict with the bypass and the degradation posture
    (degraded threshold 0.62 when the consortium is down — degradation/sources.py)."""
    pass  # fraud_verdict_code 1..4 plus the firing-based reason set

Verdict = module(bypass_eligible, verdict_and_reasons, name="precedence",
                 taps=["fraud_verdict_code"])
