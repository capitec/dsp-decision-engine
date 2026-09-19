"""Stage 5.4a -- entity criticality. Entity grain.

Three classes, and the class is the single most consequential value in the
project: it selects the event amount thresholds two grains down (s5.5), it
selects the disposition of every entity disqualification rule (s5.4), and it
gates the PP-06 caps (s5.8).

s5.4's "Records" clause is the reason this is five steps rather than one:

    "the criticality class AND the values that produced it, because 'critical
    because 25.4% effective ownership across two paths' is a different audit
    answer from 'critical because sole director', and both must be reproducible
    against a structure that may since have changed."

So the *cause* is an output, not a comment, and it is a separate value from the
class. One step per limb of the definition, then a resolve. That is five
artefacts where a single `if/elif` chain would be one -- and it is the right
trade, because the five are individually testable, individually tunable and
individually visible in the generated artefact.
"""

from decider2 import module, param, step
from grains import Entity, PERIPHERAL, SIGNIFICANT, CRITICAL, CRITICALITY_ORDER


def critical_by_ownership(
    effective_ownership_pct: float,
    critical_ownership_threshold: float = param(
        25.0, ge=0.0, le=100.0,
        description="s6.1 critical ownership threshold. Quarterly, Policy."),
) -> bool:
    """Effective ownership at or above the critical threshold."""
    pass


def critical_by_control(is_controlling: bool) -> bool:
    """Disclosed control, which may exist without ownership."""
    pass


def critical_by_sole_office(
    is_sole_director: bool,
    is_sole_trustee: bool,
    is_sole_member: bool,
) -> bool:
    """Sole director, sole trustee or sole member."""
    pass


def critical_by_surety(is_required_surety: bool) -> bool:
    """Set by policy at s5.13, not by the applicant."""
    pass


def critical_by_guarantee_share(
    guarantee_share_of_required_cover: float,
    guarantor_materiality_pct: float = param(20.0, ge=0.0, le=100.0),
) -> bool:
    """Corporate guarantor providing more than 20% of required cover."""
    pass


def significant_by_ownership(
    effective_ownership_pct: float,
    significant_ownership_threshold: float = param(
        10.0, ge=0.0, le=100.0,
        description="s6.1 significant ownership threshold. Quarterly, Policy. "
                    "Distinct from the 5.0% expansion floor and the 5.0% blend "
                    "inclusion floor -- change scenario 1 moves one of the three."),
) -> bool:
    """Effective ownership in [10%, 25%)."""
    pass


def significant_by_office(relationship_type_code: int) -> bool:
    """A director or trustee without control."""
    pass


def criticality_class(
    critical_by_ownership: bool,
    critical_by_control: bool,
    critical_by_sole_office: bool,
    critical_by_surety: bool,
    critical_by_guarantee_share: bool,
    significant_by_ownership: bool,
    significant_by_office: bool,
) -> int:
    """Resolve to one of PERIPHERAL / SIGNIFICANT / CRITICAL."""
    pass


def criticality_cause_code(
    critical_by_ownership: bool,
    critical_by_control: bool,
    critical_by_sole_office: bool,
    critical_by_surety: bool,
    critical_by_guarantee_share: bool,
    significant_by_ownership: bool,
    significant_by_office: bool,
) -> int:
    """Which limb of the definition put this entity in its class.

    Ranked, not or-ed, so it is a single stable code. All seven booleans are
    also tapped, so the pack can show that an entity was critical on three
    grounds and would still be critical if one were removed.
    """
    pass


Criticality = module(
    critical_by_ownership, critical_by_control, critical_by_sole_office,
    critical_by_surety, critical_by_guarantee_share,
    significant_by_ownership, significant_by_office,
    criticality_class, criticality_cause_code,
    name="criticality",
    grain=Entity,
    taps=["criticality_class", "criticality_cause_code",
          "critical_by_ownership", "critical_by_control",
          "critical_by_sole_office", "critical_by_surety"],
    contract="contracts/criticality.json",
)
