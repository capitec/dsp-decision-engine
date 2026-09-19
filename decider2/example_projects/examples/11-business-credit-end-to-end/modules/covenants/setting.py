"""O15 -- covenant setting. The phase where the lifecycle actually begins.

Spec 5.3 O15. Project 05 lists six covenants in four lines and monitors none of
them. This phase's output is consumed for the next five years by an entry point
(EP-4) that runs 2.4 M times a year, which makes it the highest-leverage 74
decision points in the project.

Two requirements set here and enforced at L2:

  1. Levels are set with DECLARED HEADROOM to the base case, and the headroom is
     recorded. A DSCR covenant at 1.30 against a base case of 1.31 is not a
     covenant, it is a trap. Convention: 20% cushion on cover ratios, 0.25x on
     leverage-style. A set outside the convention is an exception requiring the
     next authority level up.
  2. The definition version is frozen here and never moves.
"""

from decider2 import module, step, param, Branch
from modules.covenants.definition import bind
from time.dating import covenant_definitions

default_sets = "tables/covenants/default_sets.csv"     # 9 types x 4 grade bands x 5 amount bands


def default_covenant_set(facility_type: int, risk_grade: int, amount_cents: int) -> list:
    """180 rows. The starting set, before rule-driven additions."""
    pass  # lookup


def rule_driven_additions(gearing: float, facility_type: int, security: list) -> list:
    """Spec 5.3 O15: a director loan subordination undertaking where gearing
    exceeds 2.5; a clean-down where the facility is revolving; a borrowing base
    certificate for product 54; an LTV covenant for 53; tenancy undertakings
    where rental income services the debt; ownership-change consent above
    R2 000 000.
    """
    pass  # append by rule; each addition carries the rule id that added it


@step(description="Headroom to the base case, recorded, and an exception if thin")
def headroom_to_base_case(
    proposed_level: float, base_case: float, family: int,
    cover_cushion_pct: float = param(20.0, ge=5, le=50,
        description="Minimum cushion on cover-ratio covenants"),
    leverage_cushion_x: float = param(0.25, ge=0.05, le=1.0,
        description="Minimum cushion on leverage-style covenants"),
) -> float:
    """The cushion is a PARAMETER (Policy, annual) and the resulting level is a
    CONTRACTUAL value. That transition -- a policy parameter producing a value
    that then becomes immune to policy -- happens exactly here and nowhere else
    in the project, and it is the reason O15 is the phase worth reviewing most
    carefully. Everything downstream of this step is bound; everything upstream
    is tunable.
    """
    pass  # compute cushion; flag exception -> authority +1 if below convention


def ownership_change_baseline(known_entities: list) -> dict:
    """Spec 5.13.3 requirement 1: the baseline is stored IN THE COVENANT
    INSTANCE, not recomputed from history at test time.

    Recomputing it depends on facts learned since, which moves a baseline that
    is a contractual term. A 2029 test against a baseline recomputed in 2031
    from a register that has since been restated is a test against a different
    contract. So the shareholder register as at documentation is serialised into
    the instance, with its own knowledge date, and the test compares the
    effective-date view at the test date against that stored snapshot.
    """
    pass  # snapshot known.entities at documentation into the instance


def freeze_definitions(covenant_set: list, documented_on) -> list:
    """Bind each instance to the definition version agreed NOW. Never re-bound.

    Spec 5.7 requirement 2: an amendment INHERITS the facility's covenant
    definitions unless it explicitly replaces them, and a replacement creates a
    NEW version dated at the amendment, leaving every prior test bound to the
    prior version. So `bind` is called on new instances only, and there is no
    code path anywhere in the project that calls it twice for one instance.
    """
    pass  # for each: bind(template, version_in_force_at(documented_on), documented_on)


CovenantSetting = module(
    default_covenant_set, rule_driven_additions, headroom_to_base_case,
    ownership_change_baseline, freeze_definitions,
    name="covenant_setting", owner="covenant_operations",
    co_owners=["legal", "business_credit_risk_policy"],
    taps=["headroom_to_base_case", "branch_path"],
)
