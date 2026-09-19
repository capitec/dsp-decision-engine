"""Stage 5.13 -- FV-01..11, the conditions catalogue, and the outcome gate.

The eight consistency rules "each recompute something the flow already produced
and require agreement". That makes them a different kind of thing from every
other rule in this project: they are assertions about the flow, not about the
business, and a failure is a defect rather than a decline.

They are in the pipeline and not in `tests/` because s5.13 puts them there, and
because the ones that matter -- FV-01, FV-02, FV-04 -- can only fail on data
shapes a corpus will not contain.
"""

from decider2 import module, param, step, verdict, fires, witness, table, round_half_up
from grains import Application, Entity


def fv_01_instalment_recomputes(
    instalment: int, offered_amount: float, term_months: float,
    nominal_annual_rate: float, initiation_fee: int, monthly_service_fee: int,
) -> bool:
    """The instalment recomputes from amount, term, rate and fees TO THE CENT.

    doc 03 s1.2 is the reason this is a real check and not a formality: njit's
    `round` and CPython's `round` disagree at .xx5, and one cent of disagreement
    fails downstream reconciliation. `round_half_up` is used everywhere in this
    project and a lint enforces it; FV-01 is what catches the step that did not.
    """
    pass


def fv_02_dscr_recomputes(candidate_dscr: float, ebitda_after_haircuts: float,
                          instalment: int) -> bool:
    """The DSCR recomputes from the spread and the instalment."""
    pass


def fv_03_pricing_grade_is_final(
    risk_grade_used_for_pricing: int, risk_grade_final: int
) -> bool:
    """The grade used to price equals the final grade after any override.

    Fires when an override lands after pricing. The ordering in
    pipelines/business_facility.py makes that impossible -- BusinessGrade
    precedes EnumerateCandidates -- which is exactly the kind of thing that is
    true of the pipeline expression and checkable by reading it. FV-03 is the
    belt to that brace, and it is cheap.
    """
    pass


def fv_04_security_type_agrees(
    security_type_used_for_rate: int, security_schedule_achieved_class: int
) -> bool:
    """The security type used for the rate lookup equals the schedule's achieved
    class. The circularity in s5.11 is what makes this worth checking."""
    pass


def fv_05_within_every_ceiling(
    offered_amount: float, appetite_maximum: float, security_cover_capacity: float,
    group_exposure_headroom: float, product_maximum_amount: float,
    statement_based_cap: float,
) -> bool:
    """At or below every ceiling INDIVIDUALLY, not just below their minimum."""
    pass


def fv_06_ownership_reconciles(ownership_reconciles: bool,
                               structure_unresolved: bool) -> bool:
    """Ownership reconciles to 100% +/- 0.5%, or structure_unresolved is set."""
    pass


def fv_07_reason_codes_exist(fired_reason_mask: int, decision_date: int) -> bool:
    """Every fired reason code exists in the registry version in force at
    decision_date.

    Checked at BUILD time against every registry version in the retention
    window, not only at run time -- because a rule that can fire a code the
    registry no longer carries is a latent defect in every future replay, and
    the point at which it is cheap to find is the point the rule is written.
    """
    pass


def fv_08_no_unresolved_disqualification(
    outcome_code: int,
    business_arod_verdict: int,
    entity_disqualification_worst: int,
    entity_adverse_worst: int,
) -> bool:
    """No approval carries an unresolved disqualification at s5.3, s5.4 or s5.6.

    `entity_disqualification_worst` and `entity_adverse_worst` are Gathers over
    ALL entities -- including those PP-01 excluded from the blend. That is
    PP-07's "materiality exclusion is not an amnesty" enforced at the outcome
    rather than trusted at the blend. A 2% shareholder with a confirmed fraud
    marker reaches the outcome through this rule, and only through this rule.
    """
    pass


# --------------------------------------------------------------------------
# The outcome gate. s5.1: exceeding the structure bounds "does not decline the
# application. It produces structure_unresolved with the specific cause,
# suspends the automated outcome, and routes to credit committee with the
# partial structure and the unexpanded remainder listed. An application the Bank
# cannot see through is a referral, not a rejection -- BUT IT MAY NEVER BE AN
# APPROVAL."
#
# Three-valued, and the asymmetry is the requirement. A boolean `can_approve`
# would be enough for the machine and would lose the distinction a committee
# needs.
# --------------------------------------------------------------------------
def outcome_code(
    offer_outcome_code: int,
    business_arod_verdict: int,
    people_caps_verdict: int,
    structure_unresolved: bool,
    insufficient_people_coverage: bool,
    minimum_composition_fails: bool,
    dispute_would_change_outcome: bool,
    fv_failures: int,
) -> int:
    """Approve / approve with conditions / refer / decline.

    Five of the eight inputs can only downgrade. `structure_unresolved` and
    `dispute_would_change_outcome` downgrade an approval to a referral and leave
    a decline alone -- the second because AE-C-22 says "the Bank does not
    decline a business on a record the individual is formally contesting", and
    a decline caused by something else is not that.
    """
    pass


def referral_queue_code(
    outcome_code: int, screening_outcome_code: int, structure_unresolved: bool,
    insufficient_people_coverage: bool, sector_restricted_flag: bool,
    risk_grade_final: int, dispute_would_change_outcome: bool,
    override_within_authority: bool, jurisdiction_is_domestic: bool,
    has_business_rescue_history: bool, sector_above_portfolio_cap: bool,
) -> int:
    """The ten mandatory committee triggers, irrespective of amount."""
    pass


def authority_level_required(offered_amount: float, referral_queue_code: int) -> int:
    """Five authority levels by amount, escalated by the referral trigger."""
    pass


FinalValidation = module(
    fv_01_instalment_recomputes, fv_02_dscr_recomputes,
    fv_03_pricing_grade_is_final, fv_04_security_type_agrees,
    fv_05_within_every_ceiling, fv_06_ownership_reconciles,
    fv_07_reason_codes_exist, fv_08_no_unresolved_disqualification,
    outcome_code, referral_queue_code, authority_level_required,
    name="final_validation", grain=Application,
    taps=["outcome_code", "referral_queue_code", "authority_level_required",
          "fv_failures"],
    contract="contracts/final_validation.json",
)


# --------------------------------------------------------------------------
# Conditions precedent and covenants. A catalogue of 34, attached by rule, which
# is a `verdict(collect="all")` whose outcome is a SET rather than a severity --
# every fired rule contributes a condition and none of them wins.
#
# That is a fourth shape for the same kind, and it is the one that shows the
# kind is earning its place: `resolve="union"` instead of `resolve=SEVERITY`.
# --------------------------------------------------------------------------
CONDITIONS_CATALOGUE = table(
    "conditions_precedent",
    key="condition_code",
    columns=("description", "category", "blocks_drawdown", "owner",
             "client_facing_wording"),
    source="tables/conditions_precedent.csv",
    effective_dated=True,
    owner="Legal and Policy",
    cadence="semi-annual",
)


def cp_debtors_cession_attaches(debtors_book_contributes_to_cover: bool) -> bool:
    """Cession of debtors where the debtors book contributes to cover."""
    pass


def cp_surety_executed_attaches(
    effective_ownership_pct_max: float,
    is_sole_director_present: bool,
    surety_threshold_pct: float = param(
        25.0, ge=0.0, le=100.0,
        description="FV-09: every entity above this provides a personal surety, "
                    "unless waived by committee with a recorded rationale"),
) -> bool:
    """FV-09/FV-11."""
    pass


def cp_key_person_insurance_attaches(
    offered_amount: float,
    critical_natural_person_count: int,
    key_person_threshold: float = param(2_000_000.0, ge=0.0),
) -> bool:
    """Facility above R2m and a single critical natural person."""
    pass


ConditionsAndCovenants = verdict(
    name="conditions_and_covenants",
    of=Application,
    writes="conditions_attached",
    resolve="union",                   # every fired rule contributes
    collect="all",
    rules=[
        fires(cp_debtors_cession_attaches, gives="CP-CESSION-DEBTORS"),
        fires(cp_surety_executed_attaches, gives="CP-SURETY-EXECUTED",
              attributes=witness("entities_requiring_surety")),
        fires(cp_key_person_insurance_attaches, gives="CP-KEY-PERSON"),
        # ... thirty-one more
    ],
    interior="config/business_facility/rules/conditions.json",
)
