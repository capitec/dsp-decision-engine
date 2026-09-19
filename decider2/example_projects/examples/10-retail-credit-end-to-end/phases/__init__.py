"""The eighteen phases, as envelopes.

A PHASE is the unit this project adds above `module`.  Spec 5.1 is careful
about it -- "a requirements-level unit, not a proposed implementation unit ...
whether a phase becomes one component or forty is exactly the question this
project asks and deliberately does not answer."

This project's answer: a phase is a MODULE WITH A DECLARED ENVELOPE, and it
becomes one kernel, not forty.  Inside it there are between 31 and 205 decision
points, organised as whatever the subject wants -- a ruleset, a waterfall, a
scorecard, a search.  The envelope carries the eight things a module cannot:

    id, title       stable `phase_id` (spec 4.6: declared, never positional)
    owners          the 1..5 teams; checked against OWNERS.toml
    decision_points an ASSERTION against the registry, not a declaration
    budget          a reference into budgets/*.toml, per entry point
    degradation     which declared sources can degrade it, and to what
    basis           actual | inherited | explicit (values/bases.py)
    isolation       for P17 only (ordering.py O-18)
    emits           reasons, ceilings, record sections

Why an envelope rather than more `module(...)` kwargs: every field above is a
property of the phase AS A COMPOSED THING and is read by something other than
the compiler -- the budget by the overrun policy, the owners by CODEOWNERS
generation and blast-radius routing, the decision-point count by the registry
assertion, the basis by the build's read-checker.  Putting them on `module()`
would make `module()` the place where governance lives, which it is not.

NOTE WHAT IS ABSENT FROM EVERY ENVELOPE: `runs_on`.  Applicability is derived
(entrypoints/manifest.py).  A phase does not know which entry points it runs on
and must not, because the moment it does, "runs on 1, 2, 5, 6" is a literal
that will be right in six of the eight.

FRAMEWORK-DEMANDS #1, #4, #12, #14.
"""

from __future__ import annotations

from decider2.phase import budget_ref, phase

from degradation.sources import (
    ADJUSTMENT_REGISTER, BUREAU, CONSORTIUM, EXPOSURE_STATE, FEATURE_MART,
    FRAUD_SERVICE, IDENTITY_SERVICE, RATE_CARD, REASON_REGISTRY, STATEMENT_AGG,
)
from phases.p01_admission import routing
from phases.p03_consent_eligibility import consent, gates
from phases.p05_fraud import families, precedence
from phases.p06_features import bands, bureau_normalise, income, obligations
from phases.p09_policy_gates import register as cap_register
from phases.p10_affordability import chain, modes, verdict
from phases.p12_pricing import annuity, credit_life, fees, rate
from phases.p13_solve import search
from phases.p14_consolidation import objective, scenarios, settleability
from phases.p17_validation import assertions

# ---------------------------------------------------------------------------

P01 = phase(
    id=1, name="admission", title="Request validation and routing",
    body=routing.Normalise | routing.Validate | routing.FixDecisionDate | routing.Route,
    owners=("T1",), decision_points=41,
    budget=budget_ref("P01"),
    degradation=None,        # "None available. This phase has no external
                             # dependency by design, precisely so that it
                             # cannot fail for a reason outside the Bank."
    basis="actual",
    emits=("entry_point_code", "phase_set_id", "decision_date", "candidate_products"),
    # The one phase that exists only because there are eight entry points, and
    # the only place from which "why did phase 14 not run for this client" is
    # answerable without inference.
    produces_phase_set=True,
)

P03 = phase(
    id=3, name="consent_eligibility", title="Consent and hard eligibility",
    body=consent.SixClasses | gates.HardEligibility,
    owners=("T3", "T4"), decision_points=62,
    budget=budget_ref("P03"),
    degradation=(
        # Three sources, three different answers.  Declared per phase, per
        # source, in advance -- never computed after the fact (spec 5.25).
        ("exclusion_list_stale_6h", "continue_mark_hold_above_R50k"),
        ("sanctions_list_down",     "hard_stop_all_entry_points"),
        ("debt_review_registry_down", "continue_mark_cap_R25k"),
    ),
    basis="actual",
    # Spec 5.4's short-circuit tension, as a declaration rather than a warning:
    # every gate is EVALUATED and every verdict recorded; only the expensive
    # work downstream is short-circuited.  `evaluate_all` makes "a gate that was
    # not evaluated" and "a gate that was evaluated and passed" different facts
    # in the record by construction, which is what stops the complaint pattern
    # where a client fixes one problem, re-applies, and is declined for a second
    # problem the Bank knew about the first time.
    evaluate_all=True,
    short_circuits="downstream_only",
    emits_reasons=True,
)

P05 = phase(
    id=5, name="fraud", title="Fraud and financial crime",
    body=families.F1 | families.F2 | families.F3 | families.F4 | families.F5
         | precedence.Verdict,
    owners=("T10",), decision_points=205,
    budget=budget_ref("P05"),
    degradation=((FRAUD_SERVICE, "verdict_4_refer_except_bypass"),
                 (CONSORTIUM, "29_rules_not_evaluated_threshold_0_62")),
    basis="actual",
    evaluate_all=True,      # 188 live rules, no early exit; 7 firings across 3
                            # families is normal on a genuinely fraudulent
                            # application and all 7 are recorded
    shadow_rules=31,        # recorded, measured, governed; NOT decision points,
                            # because they provably cannot affect the answer
    emits=("fraud_verdict_code",),
    # Its verdict is read by six later phases, so it is not a gate.  Declaring
    # the consumers here is what makes "P05 is a pass/fail" a build error the
    # first time someone tries to collapse it.
    consumed_by=("p09", "p11", "p14", "p16", "p17", "p18"),
)

P06 = phase(
    id=6, name="features", title="Feature derivation",
    body=bureau_normalise.Normalise | income.Determine | obligations.Derive | bands.Band,
    owners=("T2", "T5"), decision_points=134,
    budget=budget_ref("P06"),          # 13.0 ms, second largest in the flow
    degradation=((BUREAU, "31_features_unavailable_4_scorecards_unusable"),
                 (STATEMENT_AGG, "income_falls_to_best_remaining_tier"),
                 (FEATURE_MART, "cycle_does_not_run")),
    basis="explicit",                  # it is the PRODUCER of the hypotheticals
    emits=("net_monthly_income", "living_expenses", "existing_obligations",
           "segment_code", "income_band_code", "bureau_as_of_date"),
    # Spec 5.7: the determination is performed ONCE per decision and every
    # consumer is required to consume that result.  "An implementation that
    # re-derives income inside P10 is wrong even when it happens to agree."
    # values/register.py's `produced_by` is single and exclusive, so a second
    # producer anywhere does not compile.
    exclusive_producer=True,
)

P09 = phase(
    id=9, name="policy_gates", title="Policy gates and the cap waterfall",
    body=cap_register.RegulatorySeeds | cap_register.Register
         | cap_register.ExposureAndConcentration | cap_register.PolicyGates,
    owners=("T4", "T7", "T8", "T9", "T3", "T1", "T10"),   # five-way split of 196
    decision_points=196,
    budget=budget_ref("P09"),
    degradation=((EXPOSURE_STATE, "cap_0301_on_stale_figure_hold_approvals"),
                 ("watchlist_down", "cap_0266_not_evaluated_blanket_R60k")),
    basis="actual",            # the Bank's real exposure does not fall because
                               # a scenario was evaluated.  Reaching this phase
                               # under a hypothetical ambient basis is a BUILD
                               # error, not a runtime one.
    narrows=("amount_cap", "term_cap", "limit_cap", "worst_acceptable_grade",
             "instalment_cap"),
    # O-10: the register runs in two passes, partitioned by which ceiling an
    # entry narrows.  `.pass_one` and `.pass_two` are VIEWS of one phase, one
    # params namespace and one interior document -- not two phases.
    passes=("pass_one", "pass_two"),
    pass_partition="narrows",
    emits_reasons=True,
)

P10 = phase(
    id=10, name="affordability", title="Affordability",
    body=chain.Chain | modes.SelectEvidenceMode | verdict.Verdict,
    owners=("T4", "T3"), decision_points=88,
    budget=budget_ref("P10"),
    per_reevaluation_budget_ms=1.2,   # its per-call cost is a design constraint
                                      # for a phase it does not own and cannot
                                      # see (P14).  Declared, so the negotiation
                                      # in spec 5.26.5 has a number in it.
    # "There is no degraded affordability mode, and there DELIBERATELY is not
    # one."  `degradation=REFUSED` is not the same as `degradation=None`: None
    # means nothing can degrade it, REFUSED means a degraded source reaches it
    # and the declared answer is to refuse rather than to guess.  What exists
    # instead is degraded EVIDENCE, which is a different thing with a different
    # name and a recorded evidence tier.  They are one word apart and they are
    # opposite statements (spec 5.11).
    degradation="REFUSED",
    degraded_evidence_permitted=True,
    basis="inherited",                # one arithmetic, four evidence modes,
                                      # 251 obligation versions, zero `if`s
    sites=("product_neutral", "routed", "loop", "scenario"),
    # A mode may not change the arithmetic.  It selects evidence rules,
    # parameter sets and which outputs are produced.  The build checks it: the
    # four modes share one step set and differ only in `param()` bindings and
    # `missing_as` policy.  A mode requiring a different step does not compile.
    modes_share_arithmetic=True,
    emits=("discretionary_income", "max_affordable_instalment",
           "affordability_verdict_code"),
)

P12 = phase(
    id=12, name="pricing", title="Pricing",
    body=rate.Rate | fees.Fees | credit_life.Premium | annuity.Instalment,
    owners=("T6", "T3"), decision_points=83,
    budget=budget_ref("P12"),
    per_call_budget_us=18.0,          # called as a BODY up to 400 times by P14
                                      # and 152 times by P13, by two phases that
                                      # do not own it.  Three budgets, two of
                                      # them owned by other teams.
    degradation="REFUSED",            # "no defensible way to price without a
                                      # card, and last month's card is a price
                                      # the Bank did not publish"
    basis="inherited",
    # Three rate representations in one phase is a deliberate Treasury choice
    # and must REMAIN VISIBLE AS ONE.  Declaring them means the record carries
    # which representation priced the offer, and P17 assertion 1 knows which
    # arithmetic to re-derive.  Spec 13 asks what a shared phase whose output
    # SHAPE varies by product does to the record; this is the answer.
    representations={"absolute": (10, 11, 30), "margin_over_reference": (40,),
                     "promotional_plus_reversion": (20, 21)},
    emits=("nominal_annual_rate", "initiation_fee", "monthly_service_fee",
           "credit_life_premium", "instalment", "effective_annual_rate"),
)

P13 = phase(
    id=13, name="solve", title="The solve",
    body=search.SolvePerTerm,
    owners=("T1", "T7"), decision_points=31,
    budget=budget_ref("P13"),         # 38.0 ms -- the largest single allocation
    degradation="REFUSED",            # a solve that cannot complete refers
    basis="inherited",
    # The evaluation ceiling is a CORRECTNESS parameter, not a performance one,
    # and is therefore not available to be cut when the budget is squeezed.
    # Marking it here is what makes spec 11 scenario 9 ("the latency budget is
    # cut by 40%") a negotiation with a fixed point in it.
    correctness_parameters=("evaluations_per_term",),
)

P14 = phase(
    id=14, name="consolidation", title="Consolidation search",
    body=settleability.Classify | scenarios.Generate | scenarios.Evaluate
         | objective.Rank,
    owners=("T12",), decision_points=63,
    # Three budgets, ONE implementation.  A structure in which the three become
    # three copies is a structure in which the anti-harm rule is tightened in
    # one of them next quarter and not in the other two (spec 5.15).
    budget=budget_ref("P14"),
    budget_profiles={"interactive": 820, "loop": 180, "batch": 55},
    degradation=(("settlement_quotation_provider_down", "estimate_mark_conditional"),
                 ("settlement_quotations_40pct_down", "refer")),
    basis="explicit",                 # it PRODUCES the hypotheticals
    produces_basis="hypothetical",
    max_live_scenarios=250,
)

P17 = phase(
    id=17, name="validation", title="Final validation",
    body=assertions.AssertionSet,
    owners=("T1", "T3"), decision_points=61,
    budget=budget_ref("P17"),
    degradation="REFUSED",            # "None, ever. A validation that cannot
                                      # run means the offer does not ship."
    basis="explicit",
    isolation="O-18",                 # ordering.py; a scope restriction
    abandonable=False,
    # Any mismatch is a hard failure.  Not a warning, not a logged anomaly, not
    # a value quietly corrected.  The failure rate is a monitored metric and
    # should be zero; any non-zero rate is a defect somewhere UPSTREAM.
    on_mismatch="withdraw_refer_queue_7_raise_incident",
    monitored_failure_rate_target=0.0,
)

# P02, P04, P07, P08, P11, P15, P16, P18 follow the same shape and are elided
# here for length.  Every one has an id, an owner set, a decision-point
# assertion, a budget reference, a degradation tuple and a basis posture; there
# is no phase without them, because a phase without a budget is a phase whose
# overrun is somebody else's slack.

PHASES = (P01, ..., P03, ..., P05, P06, ..., P09, P10, ..., P12, P13, P14,
          ..., P17, ...)

# Asserted at build against the registry, and against OWNERS.toml.
EXPECTED_PHASES = 18
EXPECTED_DECISION_POINTS = 1400         # excluding campaign leaves; see README 11
EXPECTED_SINGLE_OWNER_PHASES = 5        # P02, P05, P07, P13, P14
