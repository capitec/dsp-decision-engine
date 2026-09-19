"""The fourteen policy interventions, and rejection reasons for things that did not happen.

These fire DURING evaluation and constrain it. A violated intervention
INVALIDATES the scenario - it does not adjust it, score it down, or refer it.

Five requirements on the mechanism, from spec 5.7, and each one shapes the code:

  1. all violations are evaluated, not only the first. A scenario rejected by
     four interventions is a different conversation from one rejected by a single
     marginal breach, and the second kind is the one a consultant may usefully
     escalate. So no short-circuit. Fourteen comparisons, every time, on every
     row.

  2. every violation carries its ACTUAL AND THRESHOLD values. "Rejected by
     CON-INT-04: total cost +23.4% against a threshold of 15%" is an answer.
     "Rejected by policy" is not, and the contact centre will ask, because the
     client will ask.

  3. where an overlay changed a threshold, the record names the overlay, its
     approval reference, its effective window and the base value it replaced.

  4. NOT-APPLICABLE IS DISTINCT FROM PASSED. CON-INT-10 does not apply to product
     20. A record showing it as passed is misleading.

  5. interventions must be evaluable in isolation, so Credit Risk Policy can ask
     "how many of last month's assessments would have changed if CON-INT-04 moved
     to 12%" without re-running the world by hand.


WHERE THE REJECTIONS LIVE, which is spec question 11.

Not in objects. The evaluated scenarios are a 400-row frame, so the rejections
are COLUMNS OF THAT FRAME: fourteen verdict columns (int8: PASS/FAIL/NA), and for
each intervention an actual and a threshold column. That is 42 columns over 400
rows - about 130 KB per assessment before compression, 780 MB a day at 6 000
assessments, and it writes as one parquet file keyed on application_id.

A contact centre agent's "why did you not settle my furniture account" is then a
filter on one file, answered in milliseconds, which is spec 9.5's "within
seconds, not on request to a data team". Retention drops the detail at three
years and keeps counts by reason, and the retention period is a parameter because
Compliance will change it.

This is the same mechanism as doc 04 4.1's taps. What changes is WHAT A RECORD
IS. Taps are per-record diagnostic columns where a record is a client; here a
record is a HYPOTHESIS ABOUT A CLIENT, and there are 400 of them. The framework
does not currently distinguish those. See FRAMEWORK-DEMANDS D9.
"""

from decider2 import module, param, step
from decider2.policy import Verdict, intervention
from decider2.values import Maybe, na
from decider2.params import overlayable


# --- the declaration form -----------------------------------------------------
#
# `@intervention` is `@step` with three additions that every one of the fourteen
# needs and that nothing else in the framework provides:
#
#   id=          the join key into the rejection reason registry (~60 codes x 9
#                attributes, owned by Compliance, versioned like the decline
#                taxonomy, and needed in three languages - change scenario 15).
#   applies_when= the NOT-APPLICABLE predicate. Returning na() from the body
#                would work; declaring it means the record can distinguish "did
#                not apply" from "applied and passed" WITHOUT executing, which is
#                what makes requirement 5 (evaluable in isolation) possible.
#   records=     actual and threshold, by name. Not optional, not a convention.
#
# Writing these as fourteen plain @step functions would work and would lose
# requirement 2 within a quarter, because nothing would force the actual and the
# threshold out of the comparison and into the record.


@intervention(
    id="CON-INT-01",
    description="Maximum accounts settled in one consolidation.",
    records=("accounts_settled", "max_accounts"),
)
def max_accounts_settled(
    accounts_settled: int,
    max_accounts: int = param(8, ge=3, le=15),
) -> Verdict:
    """Per product. Eight on 11 and 30, six on 20, ten on 40."""
    pass  # PASS if accounts_settled <= max_accounts else FAIL


@intervention(
    id="CON-INT-02",
    description="No settlement of an account opened within N months.",
    records=("youngest_settled_account_months", "min_account_age_months"),
)
def no_recent_account(
    youngest_settled_account_months: int,
    min_account_age_months: int = param(3, ge=0, le=12),
) -> Verdict:
    """Global. Also applied at settleability (class 5), and deliberately twice.

    Settleability removes the account from CANDIDACY; this removes a SCENARIO
    that somehow contains one. The duplication is the point: a change to the
    settleability rule that accidentally let a three-week-old account through
    would otherwise reach a client, and this is the net under it. Two evaluations
    of the same threshold from one parameter, which doc 01 5.3's law permits
    because there is still exactly one canonical location for the value.
    """
    pass  # PASS if youngest >= min else FAIL


@intervention(
    id="CON-INT-03",
    description="New instalment at least X% below the sum of settled instalments.",
    records=("instalment_relief_pct", "min_relief_pct"),
)
def min_instalment_relief(
    instalment_relief_pct: float,
    min_relief_pct: float = param(10.0, ge=0.0, le=30.0),
) -> Verdict:
    """Per product, per channel. On product 20 this is measured against the
    STRESSED payment, which is why the card team holds the floor at 0: the
    stressed payment is already the binding test and a second floor over the same
    figure is a threshold nobody can reason about.
    """
    pass  # PASS if relief >= floor else FAIL


@intervention(
    id="CON-INT-04",
    description="ANTI-HARM. New total cost of credit may not exceed the settled "
                "accounts' remaining cost by more than Y%.",
    records=("total_cost_uplift_pct", "max_cost_uplift_pct"),
)
def anti_harm(
    total_cost_of_credit: float,
    settled_remaining_cost_total: float,
    max_cost_uplift_pct: float = overlayable(15.0, scope=["product_code", "channel_code"]),
) -> Verdict:
    """The rule the whole flow exists to not break, and the one most of section 9
    of the spec exists because somebody will allege was not applied.

    Lengthening a term always lowers an instalment and almost always raises the
    total cost of credit. A consolidation flow that optimises the instalment
    alone is a machine for manufacturing ombud complaints. This is the rule that
    stops it, and it is four lines.

    THREE THINGS THAT DEFEAT IT IF THEY ARE GOT WRONG ELSEWHERE, none of them in
    this file:
      - product 30's total cost must include the balloon;
      - product 40's horizon must be the 84-month sub-term, not 240;
      - product 20's total cost must use the reversion rate after the promotion.
    Each is one addition in one product arm. Each, omitted, makes this
    intervention pass a scenario it should reject, silently, forever.

    `overlayable(...)`: ADJ-AH-012 sets this to 12.0 on product 11, branch
    channel, for Q3 (change scenario 2). `params.base.max_cost_uplift_pct` stays
    15.0 and both land in the record, because a scenario that would have passed
    at the base threshold and failed at the overlaid one is the single most
    likely subject of a later query - from Credit Committee as often as from a
    client.
    """
    pass  # uplift = (tcoc - settled_cost) / settled_cost * 100; PASS if <= max


@intervention(
    id="CON-INT-05",
    description="Maximum term extension over the longest settled account's remaining term.",
    applies_when="term_months is not na",
    records=("term_extension_months", "max_extension_months"),
)
def max_term_extension(
    term_months: Maybe[int],
    longest_settled_remaining_term: int,
    max_extension_months: int = param(24, ge=0, le=60),
) -> Verdict:
    """NOT APPLICABLE on product 20, and this is where that matters most.

    `applies_when` declares it. The alternative - a sentinel term of 0 - computes
    0 - 31 = -31, compares it to +24, and PASSES. A rule intended to stop term
    extension would silently approve every balance transfer, and the record would
    show CON-INT-05 as passed, and nobody would ever look again.

    The signature declares `Maybe[int]` and the framework refuses to compile a
    step that reads a may-be-na column without declaring it. That refusal is the
    whole mechanism. See FRAMEWORK-DEMANDS D4.
    """
    pass  # NA if term is na; else PASS if (term - longest) <= max


@intervention(
    id="CON-INT-06",
    description="Rate ceiling: new rate may not exceed the settled accounts' "
                "balance-weighted average rate.",
    records=("nominal_annual_rate", "rate_ceiling"),
)
def rate_ceiling(
    nominal_annual_rate: float,
    settled_weighted_rate: float,
    mode: str = param("binding", enum=["binding", "off", "plus_bps"]),
    plus_bps: float = param(0.0, ge=0.0, le=500.0),
) -> Verdict:
    """On product 20 this reads the REVERSION rate, never the promotional rate.

    The normalised `nominal_annual_rate` the card arm publishes is time-weighted
    over 36 months, which is the right figure for ranking and the WRONG figure
    for this ceiling: a 6-month promotion at 9.9% drags the normalised rate under
    a ceiling the client will be over from month seven. So the card arm publishes
    both and this intervention binds to `reversion_rate` on product 20 via the
    intervention set's own `.at()` rebind in the composition below.

    That rebind is the single most subtle thing in this file, and it is VISIBLE -
    one line in a composition expression, not a branch inside a body.
    """
    pass  # NA if mode == off; else PASS if rate <= weighted + plus_bps/100


@intervention(
    id="CON-INT-07",
    description="No consolidation for a client under debt review.",
    records=("debt_review_status", "debt_review_source"),
)
def no_debt_review(debt_review_status: int, debt_review_source: int) -> Verdict:
    """Not tunable. Global. And it is a ROUTE, not a decline.

    A client under debt review may not take new credit but their existing
    agreements CAN be restructured through the debt counsellor or court-ordered
    path. Emitting a decline is wrong, and it is what a flow that treats debt
    review as a decline gate will do (AC 11).

    The source and date are recorded because the bureau and the regulator's
    register disagree often enough that the discrepancy is itself a finding.
    """
    pass  # FAIL with route_to=RESTRUCTURE_DEBT_COUNSELLOR, never a decline code


@intervention(
    id="CON-INT-08",
    description="New money as a proportion of the settlement total, and in absolute terms.",
    records=("new_money", "new_money_ceiling"),
)
def new_money_ceiling(
    new_money_released: float,
    settlement_total: float,
    max_pct: float = param(25.0, ge=0.0, le=50.0),
    max_abs: float = param(50000.0, ge=0.0, le=150000.0),
) -> Verdict:
    """Per product, per channel. The binding one of the two is recorded."""
    pass  # PASS if new_money <= min(settlement * pct/100, max_abs)


@intervention(
    id="CON-INT-09",
    description="Post-consolidation debt service ratio ceiling.",
    records=("post_dsr_pct", "max_dsr_pct"),
)
def dsr_ceiling(
    committed_monthly: float,
    existing_obligations_after: float,
    net_monthly_income: float,
    max_dsr_pct: float = overlayable(45.0, scope=["product_code", "risk_grade"]),
) -> Verdict:
    """Per product, per grade. ADJ-DSR-009 sets it to 40% for grades 8-12 on
    product 11 this quarter - a cut-off shift, order 20, expiring 2026-09-30.
    """
    pass  # PASS if (committed + obligations_after) / income * 100 <= max


@intervention(
    id="CON-INT-10",
    description="Minimum proportion of the advance paid to external creditors.",
    applies_when="exposure_basis_code == ADVANCE",
    records=("external_proportion_pct", "min_external_pct"),
)
def min_external_proportion(
    settlement_total_external: float,
    required_advance: float,
    min_external_pct: float = param(60.0, ge=0.0, le=100.0),
) -> Verdict:
    """NOT APPLICABLE to product 20, which advances nothing - it approves a limit.

    Spec 5.7.6 names this exact intervention as the example, and `applies_when`
    is keyed on `exposure_basis_code` rather than on `product_code` so that
    adding product 21 (Access Facility, also a limit) needs no edit here.
    Predicating on the PROPERTY rather than on the product is what makes AC 15
    ("adding a fifth product does not require changes to the interventions") true
    rather than aspirational.
    """
    pass  # NA if basis is LIMIT; else PASS if external / advance * 100 >= min


@intervention(
    id="CON-INT-11",
    description="No settlement of an account in dispute.",
    records=("disputed_accounts_in_set", "threshold"),
)
def no_disputed_account(disputed_accounts_in_set: int) -> Verdict:
    """Not tunable. Also enforced at settleability (class 6). Same net as CON-INT-02."""
    pass  # PASS if zero


@intervention(
    id="CON-INT-12",
    description="Secured accounts may only be settled where the security releases or transfers.",
    records=("unreleased_secured_in_set", "threshold"),
)
def security_must_release(unreleased_secured_in_set: int) -> Verdict:
    """Not tunable. Also H4 in the ordering set, as a constraint rule.

    H4 stops such accounts entering the candidate pool; this stops a scenario
    containing one being priced. Two places, one fact, and neither is redundant:
    H4 is Credit Risk Policy's to revise in an interior document and this is not.
    """
    pass  # PASS if zero


@intervention(
    id="CON-INT-13",
    description="Maximum consolidations per client per rolling 24 months.",
    records=("consolidations_24m", "max_consolidations"),
)
def max_consolidations(
    consolidations_24m: int,
    max_consolidations: int = param(2, ge=1, le=3),
) -> Verdict:
    """Serial consolidation is the strongest predictor of the next default and
    the loudest signal in an ombud file.
    """
    pass  # PASS if consolidations_24m < max


@intervention(
    id="CON-INT-14",
    description="Minimum post-consolidation discretionary income after the new instalment.",
    records=("post_discretionary_income", "min_discretionary"),
)
def min_discretionary(
    discretionary_income: float,
    min_discretionary: float = param(850.0, ge=500.0, le=2500.0),
) -> Verdict:
    """Per product, per household size."""
    pass  # PASS if discretionary >= min


# --- the set ------------------------------------------------------------------
#
# Composition order is irrelevant - a module interior is a pure DAG (doc 03 3.1)
# and nothing here reads anything else here. That is a property worth having: it
# means adding CON-INT-15 cannot change CON-INT-04's answer, which is what makes
# requirement 5 (evaluable in isolation) structurally true rather than a claim.
#
# The one `.at()` is CON-INT-06 on product 20, and it is visible here rather than
# hidden in a branch inside the step.

PolicyInterventions = module(
    max_accounts_settled,
    no_recent_account,
    min_instalment_relief,
    anti_harm,
    max_term_extension,
    rate_ceiling.at(inputs={"nominal_annual_rate": "reversion_rate"}, when="product_code == 20"),
    no_debt_review,
    new_money_ceiling,
    dsr_ceiling,
    min_external_proportion,
    no_disputed_account,
    security_must_release,
    max_consolidations,
    min_discretionary,
    name="interventions",
    params="config/policy/interventions.json",
    taps=["verdict@*", "actual@*", "threshold@*"],
    contract="contracts/interventions.json",
)


@step(output="viability_verdict")
def viability(intervention_verdicts: list, product_rejection_codes: list) -> int:
    """VIABLE if every intervention passed or did not apply AND no product rejection.

    Note the asymmetry with how the verdicts are recorded: the verdict is a
    single value, the REASONS are all of them. A scenario rejected by four
    interventions and one rejected by one both come out non-viable, and the
    record keeps the difference because that difference is what a consultant
    escalates on.
    """
    pass  # all(v in (PASS, NA)) and not product_rejection_codes
