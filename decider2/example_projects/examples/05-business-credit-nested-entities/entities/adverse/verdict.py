"""Stage 5.6b -- the entity adverse verdict. Entity grain.

After `EntityAdverseFacts`, the twelve AE-R rules are ordinary predicates over
ordinary scalars. There is no collection here. That is the payoff of the gather:

    **The roll-up is not a component kind. It is a Gather plus a rule set.**

Spec s13 Q2 asks "is roll-up a core component kind in its own right, or is it a
rule set that happens to run over a collection?" This sketch answers: neither.
It is a *fold* (new, and the only new thing) composed with a *rule set* (which
the framework needs anyway for s5.3 and s5.4). Splitting it that way is what
makes the attribution uniform and the ordering independence structural.
"""

from decider2 import module, param, step, verdict, fires, witness, union
from grains import Entity, IMMATERIAL, MINOR, MATERIAL, DISQUALIFYING, SEVERITY_ORDER
from grains import PERIPHERAL, SIGNIFICANT, CRITICAL


# --------------------------------------------------------------------------
# Predicates. One per rule, named for the rule, reading gathered scalars.
# --------------------------------------------------------------------------

def ae_r_01_fires(disqualifying_count: int) -> bool:
    """AE-R-01: any event disqualifying."""
    pass


def ae_r_02_fires(
    minor_recent_count: int,
    minor_recent_n: int = param(
        3, ge=1, le=20,
        description="s6.1 minor-event count. Policy, quarterly and after loss "
                    "events. The window it pairs with is in flags.py."),
) -> bool:
    """AE-R-02: three or more minor events inside the count window."""
    pass


def ae_r_03_fires(
    minor_total_count: int,
    minor_total_n: int = param(5, ge=1, le=40),
) -> bool:
    """AE-R-03: five or more minor events of any age."""
    pass


def ae_r_04_fires(
    material_24m_count: int,
    criticality_class: int,
    material_n: int = param(2, ge=1, le=20),
) -> bool:
    """AE-R-04: two or more material events within 24 months, entity critical."""
    pass


def ae_r_05_fires(
    material_24m_count: int,
    criticality_class: int,
    material_n: int = param(2, ge=1, le=20),
) -> bool:
    """AE-R-05: same count, entity significant or peripheral. Caps the business
    outcome at refer -- see the `caps=` clause below."""
    pass


def ae_r_06_fires(
    unsatisfied_total: float,
    requested_amount: float,                      # broadcast from Application
    aggregate_floor: float = param(150_000.0, ge=0.0),
    aggregate_pct_of_request: float = param(15.0, ge=0.0, le=100.0),
) -> bool:
    """AE-R-06: aggregate unsatisfied above max(R150 000, 15% of requested)."""
    pass


def ae_r_07_fires(
    unsatisfied_total: float,
    requested_amount: float,
    aggregate_floor: float = param(500_000.0, ge=0.0),
    aggregate_pct_of_request: float = param(50.0, ge=0.0, le=100.0),
) -> bool:
    """AE-R-07: aggregate unsatisfied above max(R500 000, 50% of requested)."""
    pass


def ae_r_08_fires(material_recent_count: int) -> bool:
    """AE-R-08: any material event inside the recency window.

    Floors the verdict at material and sets `recent_adverse`, which blocks
    entity grades 1-3 in entities/scoring/families.py and feeds PP-06's
    "recent_adverse on any critical entity" cap.
    """
    pass


def ae_r_09_fires(
    dishonoured_12m_count: int,
    velocity_n: int = param(
        8, ge=1, le=40,
        description="s6.1 velocity threshold: dishonoured payments in 12 months"),
) -> bool:
    """AE-R-09: eight or more dishonoured payments within 12 months."""
    pass


def ae_r_10_fires(
    trailing_12m_count: int,
    preceding_12m_count: int,
    trend_delta_n: int = param(3, ge=1, le=20),
) -> bool:
    """AE-R-10: trailing count exceeds the preceding count by three or more.

    Escalates the otherwise-computed verdict by one class rather than giving a
    verdict of its own. That is `escalates=1` below, and it is why the verdict
    kind needs a modifier concept and not only a severity -- see
    FRAMEWORK-DEMANDS D11.
    """
    pass


def ae_r_11_fires(
    immaterial_count: int,
    event_count: int,
    clear_count_ceiling: int = param(2, ge=0, le=10),
) -> bool:
    """AE-R-11: all events immaterial and count at or below the ceiling."""
    pass


def ae_r_12_fires(criticality_class: int, has_write_off_or_fraud: bool) -> bool:
    """AE-R-12: entity peripheral -- verdict capped at material, except where
    AE-C-18 or AE-C-20 fired."""
    pass


# --------------------------------------------------------------------------
# The verdict.
#
# Every `fires(...)` carries three attribution clauses and they are the whole of
# s5.6's "Attribution is mandatory and structured":
#
#   attributes=  the witness set -- WHICH events satisfied this rule
#   quantity=    the computed quantity where the rule is quantitative
#   gives/floors/caps/escalates=  what the rule does to the verdict
#
# `collect="all"` records every fired rule with its own attribution, which is
# what makes s5.6's "the next-most-severe rule that fired, and its events,
# because a committee routinely asks 'and if that one were removed?'" a lookup
# rather than a re-run.
#
# Note AE-R-06 and AE-R-07 share a witness. Both attribute to *every*
# contributing event, which is s5.6 requirement 2 ("Six unsatisfied judgments of
# R30 000 each are a R180 000 problem and none of them is individually
# material") and is precisely the case a worst-of roll-up cannot express.
# --------------------------------------------------------------------------
EntityAdverseVerdict = verdict(
    name="entity_adverse_verdict",
    of=Entity,
    writes="entity_adverse_verdict_code",
    resolve=SEVERITY_ORDER,
    collect="all",
    rules=[
        fires(ae_r_01_fires, gives=DISQUALIFYING,
              attributes=witness("disqualifying_count")),

        fires(ae_r_02_fires, gives=MATERIAL,
              attributes=witness("minor_recent_count"),
              quantity="minor_recent_count"),

        fires(ae_r_03_fires, gives=MATERIAL,
              attributes=witness("minor_total_count"),
              quantity="minor_total_count"),

        fires(ae_r_04_fires, gives=DISQUALIFYING,
              attributes=witness("material_24m_count"),
              quantity="material_24m_count"),

        fires(ae_r_05_fires, gives=MATERIAL,
              attributes=witness("material_24m_count"),
              quantity="material_24m_count",
              caps="business_outcome_at_refer"),

        fires(ae_r_06_fires, gives=MATERIAL,
              attributes=witness("unsatisfied_total"),
              quantity="unsatisfied_total"),

        fires(ae_r_07_fires, gives=DISQUALIFYING,
              attributes=witness("unsatisfied_total"),
              quantity="unsatisfied_total"),

        fires(ae_r_08_fires, floors=MATERIAL,
              attributes=witness("material_recent_count"),
              sets="recent_adverse"),

        fires(ae_r_09_fires, gives=MATERIAL,
              attributes=witness("dishonoured_12m_count"),
              quantity="dishonoured_12m_count"),

        fires(ae_r_10_fires, escalates=1,
              attributes=union(witness("trailing_12m_count"),
                               witness("preceding_12m_count")),
              quantity="trend_delta"),

        fires(ae_r_11_fires, gives=IMMATERIAL,
              attributes=witness("immaterial_count")),

        fires(ae_r_12_fires, caps=MATERIAL,
              unless="has_write_off_or_fraud"),
    ],

    # Composition order for the three modifier kinds. Declared, because
    # cap-then-escalate and escalate-then-cap give different answers and s5.6
    # does not say which. This is the project's decision and it is in one place.
    modifier_order=("gives", "escalates", "floors", "caps"),

    interior="config/business_facility/rules/entity_adverse_verdict.json",
    contract="contracts/entity_adverse_verdict.json",
    taps=["entity_adverse_verdict_code", "binding_rule_id", "fired_rule_mask"],
)


# --------------------------------------------------------------------------
# s5.6's last paragraph: where any contributing event was classified against an
# overlaid threshold, the verdict carries `verdict_overlay_sensitive` AND the
# verdict the unoverlaid thresholds would have produced.
#
# The second half is not computable here and must not be faked. It is a Shadow
# of ClassifyEvent + EntityAdverseVerdict with overlay position 1 disabled, and
# it is declared in grade/overlay_stack.py as part of `UnadjustedSpine`. Writing
# an approximation here -- "if the threshold moved by X, the verdict would have
# been Y" -- is the failure mode s5.6 is guarding against, because it is right
# often enough to be trusted and wrong on exactly the cases a committee reads.
# --------------------------------------------------------------------------
def verdict_overlay_sensitive(any_overlaid_threshold: bool) -> bool:
    """Any contributing event was classified against an overlaid threshold."""
    pass


def verdict_provisional(any_provisional: bool) -> bool:
    """Any contributing event is disputed. Forces the dispute shadow to govern."""
    pass


VerdictFlags = module(
    verdict_overlay_sensitive, verdict_provisional,
    name="verdict_flags", grain=Entity,
)
