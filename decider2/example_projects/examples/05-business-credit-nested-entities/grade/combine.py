"""Stage 5.10 -- the combined business grade.

Four components, weights from an 18-cell table, blending on log-odds, and the
distinction between an override and an overlay that s5.10 spends a paragraph on
because they get confused.
"""

from decider2 import module, param, step, table, Branch, overlay
from grains import Application

BLEND_WEIGHTS = table(
    "business_component_weights",
    key=("segment_code", "financial_confidence_band"),
    columns=("financial_weight", "people_weight", "behavioural_weight"),
    source="tables/component_weights.csv",
    effective_dated=True,
    owner="Business Credit Risk Policy",
    cadence="quarterly",
)

OVERRIDE_REASONS = table(
    "override_reasons",
    key="override_reason_code",
    columns=("description", "direction_allowed", "requires_second_signature"),
    source="tables/override_reasons.csv",
    effective_dated=True,
)

OVERRIDE_AUTHORITY = table(
    "override_authority",
    key="authority_level_code",
    columns=("max_facility_amount", "max_notches"),
    source="tables/override_authority.csv",
    effective_dated=True,
)


def segment_code(
    trailing_turnover: float,
    micro_ceiling: float = param(3_000_000.0, ge=0.0),
    small_ceiling: float = param(30_000_000.0, ge=0.0),
    medium_ceiling: float = param(150_000_000.0, ge=0.0),
) -> int:
    """Micro / small / medium by trailing turnover. Above medium is out of scope."""
    pass


def behavioural_available(months_of_conduct: float,
                          conduct_floor: float = param(6.0, ge=0.0, le=24.0)) -> bool:
    """BUS-BEH-01 needs six months of internal conduct."""
    pass


def component_weights_applied(
    segment_code: int,
    financial_confidence_code: int,
    behavioural_available: bool,
) -> float:
    """Reads the 18-cell table and redistributes the behavioural weight pro rata
    across the other two where the behavioural component is absent.

    Emitted as three separate values (financial_weight_applied,
    people_weight_applied, behavioural_weight_applied) because s5.10's "Emits"
    requires "every component's PD and weight in both adjusted and unadjusted
    form", and a weight that only exists inside an expression cannot be emitted.
    """
    pass


def business_log_odds(
    financial_pd: float, people_pd: float, behavioural_pd: float,
    financial_weight_applied: float,
    people_weight_applied: float,
    behavioural_weight_applied: float,
) -> float:
    """Weighted blend on log-odds, as at PP-05."""
    pass


def probability_of_default(business_log_odds: float) -> float:
    """Business PD. Overlay positions 6 and 7 apply to this value."""
    pass


def risk_grade(probability_of_default: float, product_code: int, segment_code: int) -> int:
    """core.risk_grade. Overlay position 8 shifts the boundaries."""
    pass


# --------------------------------------------------------------------------
# The qualitative override. s5.10: "An override is not an overlay. They are
# adjacent enough to be confused and must not be: an override is one analyst's
# judgement about one application, recorded against that application; an overlay
# is an approved policy instrument applying to a declared population until it
# expires. They have different authorities, different lifetimes, different
# scopes and different evidence obligations. Both may act on the same grade, and
# the record must show which did what."
#
# So the override is NOT `overlay.position(...)`. It is an ordinary input --
# the analyst's request arrives on the application -- validated by ordinary
# steps, applied by an ordinary step, and recorded separately. The framework
# keeps them apart by construction because they are different mechanisms, not
# because of a naming convention.
# --------------------------------------------------------------------------
def override_within_authority(
    requested_override_notches: int,
    authority_level_code: int,
    requested_amount: float,
    has_second_signature: bool,
) -> bool:
    """The four-level authority table, plus: improving overrides require a
    second signature, worsening overrides never do."""
    pass


def override_attempts_reversal(
    requested_override_notches: int,
    business_arod_verdict: int,
    entity_disqualification_present: bool,
    entity_adverse_disqualifying_present: bool,
) -> bool:
    """s5.10: an override may not reverse a disqualification at s5.3, s5.4 or
    s5.6 -- "those are absolute, and an override that attempts it is REJECTED
    WITH A REASON, not silently ignored"."""
    pass


def risk_grade_final(
    risk_grade: int,
    requested_override_notches: int,
    override_within_authority: bool,
    override_attempts_reversal: bool,
) -> int:
    """The grade that prices, after a valid override."""
    pass


BusinessGrade = module(
    segment_code, behavioural_available, component_weights_applied,
    business_log_odds, probability_of_default, risk_grade,
    override_within_authority, override_attempts_reversal, risk_grade_final,
    name="business_grade", grain=Application,
    taps=["segment_code", "financial_weight_applied", "people_weight_applied",
          "behavioural_weight_applied", "probability_of_default",
          "risk_grade", "risk_grade_final", "override_within_authority",
          "override_attempts_reversal", "branch_path"],
    contract="contracts/business_grade.json",
)
