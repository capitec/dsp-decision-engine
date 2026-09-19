"""The thirteen declared degradation sources. Spec 5.25. The degradation
POLICY artefact: each source's failure behaviour is declared here, in
advance, once — never computed after the fact. "It degraded gracefully" is
not an acceptable answer to "what did it do" (spec 5.25's opening line).

Every phase envelope in phases/__init__.py references these names in its
`degradation=` tuple rather than inventing its own per-phase source. That is
the difference between thirteen sources and eighteen phases each guessing —
a source's behaviour is declared exactly once and every consumer shares it.

`degraded_mode_code` (values/register.py-adjacent; carried on every record,
spec 4.6) is the COMPOSITE of whichever of these fired on one decision. Never
a single source's code standing in for the whole picture: a decision degraded
by both a stale bureau and a stale exposure figure carries both codes in
`source_degradation_codes`, and the composite is what `degraded_mode_code`
names.
"""

from __future__ import annotations

from decider2.degradation import DegradedSource

BUREAU = DegradedSource(
    "bureau_down", code=11, entry_points=(1, 2, 5), continues=True,
    behaviour="Approve only within a reduced envelope: <= R25 000, term <= 36, "
              "segments 3-5 only, internal tenure >= 24 months, no arrears in 24. "
              "Everyone else refers. Four scorecards unusable; segments 2-5 fall "
              "to SC-A1 with a declared -45 point shift.",
)

STATEMENT_AGG = DegradedSource(
    "statement_aggregator_down", code=12, entry_points=(1, 5), continues=True,
    behaviour="Income falls to the best remaining tier, usually 7 at a 35% "
              "haircut -- a real, correctly-applied penalty. Amounts above "
              "R80 000 refer.",
)

FRAUD_SERVICE = DegradedSource(
    "fraud_service_timeout", code=13, entry_points=(1, 2, 5, 6), continues=True,
    behaviour="fraud_verdict_code becomes 4 (could not be established). The "
              "bypass path (families.py's bypass_eligible) still proceeds; "
              "everything else refers, queue 3. Never approves on an unknown "
              "fraud verdict.",
)

CONSORTIUM = DegradedSource(
    "consortium_intelligence_down", code=21, entry_points=(1, 2, 5, 6), continues=True,
    behaviour="29 of 188 fraud rules recorded as NOT EVALUATED, distinct from "
              "did-not-fire. Weighted threshold drops to 0.62 to compensate.",
)

IDENTITY_SERVICE = DegradedSource(
    "identity_verification_down", code=14, entry_points=(1, 5), continues=True,
    behaviour="Confidence floor rises to 0.94; queue 2 referrals rise from "
              "0.6% to an expected 4.1%.",
)

RATE_CARD = DegradedSource(
    "rate_card_version_missing", code=31, entry_points=(1, 4, 5, 6, 7), continues=False,
    scope="per_product",
    behaviour="The affected product is withdrawn from routing and the "
              "withdrawal is recorded. Other products proceed unaffected. "
              "There is no defensible way to price without a card.",
)

STATUTORY_TABLE = DegradedSource(
    "statutory_ceiling_or_fee_schedule_missing", code=32, entry_points="all", continues=False,
    behaviour="Hard stop. Lending at an unverified rate is unlawful, not degraded.",
)

ADJUSTMENT_REGISTER = DegradedSource(
    "adjustment_register_unreachable", code=41, entry_points="all_but_7", continues=True,
    grace_minutes=90,
    behaviour="The last successfully resolved register version is used, "
              "pinned and recorded, for up to 90 minutes. Running 'unadjusted' "
              "is NOT a fallback -- not knowing whether an overlay applies is "
              "different from knowing that none does. Hard stop after 90 "
              "minutes.",
)

REASON_REGISTRY = DegradedSource(
    "reason_registry_unreachable", code=42, entry_points="all", continues="partly",
    behaviour="Approvals proceed. Declines cannot be issued -- a decline "
              "without a registered code cannot be explained to the person it "
              "declined -- so they refer instead.",
)

EXPOSURE_STATE = DegradedSource(
    "internal_account_state_stale", code=51, entry_points=(1, 2, 5, 6), continues=True,
    stale_after_hours=6,
    behaviour="CAP-0301 runs on the stale figure; approvals it would have "
              "bound are held for re-check. Increases on entry point 2 capped "
              "at R5 000.",
)

FEATURE_MART = DegradedSource(
    "feature_mart_stale", code=52, entry_points=(3, 4), continues=False,
    stale_after_hours=36,
    behaviour="The cycle does not run. A limit programme on month-old "
              "behaviour is worse than no limit programme.",
)

EVIDENCE_STORE = DegradedSource(
    "evidence_store_unavailable", code=61, entry_points="all", continues=True,
    behaviour="The decision completes; the record buffers locally; the "
              "failure is counted, alerted and reconciled monthly. Evidence "
              "capture must never be on the critical path for availability.",
)

AFFORDABILITY_TABLES = DegradedSource(
    "affordability_tables_unavailable", code=33, entry_points="all_but_7", continues=False,
    behaviour="No degraded affordability mode, deliberately. A guessed "
              "affordability answer is the basis of a reckless-lending "
              "defence.",
)

SOURCES = (BUREAU, STATEMENT_AGG, FRAUD_SERVICE, CONSORTIUM, IDENTITY_SERVICE,
          RATE_CARD, STATUTORY_TABLE, ADJUSTMENT_REGISTER, REASON_REGISTRY,
          EXPOSURE_STATE, FEATURE_MART, EVIDENCE_STORE, AFFORDABILITY_TABLES)

# Spec 5.25.1's three rules, checked at build against every source above:
#   1. degrade toward refer, never toward decline (`continues` never yields a
#      decline outcome that a normal run would not have produced);
#   2. every decision under a fired source carries degraded_mode_code and
#      source_degradation_codes as at read time;
#   3. every degraded approval/referral is re-assessable within 5 business
#      days -- budgets/overrun.py's queue, not this file's job to run.
EXPECTED_DEGRADED_SOURCES = 13
NO_DEGRADED_MODE_AT_ALL = (STATUTORY_TABLE, RATE_CARD, AFFORDABILITY_TABLES,
                           ADJUSTMENT_REGISTER)  # the last, only after 90 minutes
