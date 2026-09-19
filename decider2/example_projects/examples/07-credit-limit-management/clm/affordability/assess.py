"""The statutory assessment, and the staleness routing on top of it (s5.6).

`core.affordability` is consumed here exactly as project 02 consumes it. The
three things that differ -- the notional instalment, the wider buffer, and the
staleness tolerances -- are all inputs.
"""

from decider2 import Branch, module, panel, param
from decider2.credit import core, overlay_point


def notional_instalment_c(proposed_limit_c: int, product_code: int, balance_band: int,
                          tables) -> int:
    """A revolving facility has no instalment, so one is imputed: the commitment
    the client would carry if the facility were drawn to the new limit and
    serviced at the contractual minimum payment rate (3.5% card, 5.0% facility).
    A R25 000 card limit carries a R875 notional instalment."""
    pass


ApplyBufferOverlays = overlay_point(
    "buffer_overlays",
    register="overlay_set",
    adjusts={"affordability_buffer": "percentage_points"},
    scope_keys=["product_code", "employment_type_code", "evidence_tier_code"],
    order="declared",
)

# The shared capability. Same object project 02 composes; a different buffer.
# 18% here against origination's 12%, because origination underwrites against
# verified evidence collected days earlier and this programme underwrites
# against a salary pattern and a bureau file.
Affordability = core.affordability.at(
    inputs={"instalment": "notional_instalment_c"},
).bind(affordability_buffer=0.18)


# --- the staleness rule ----------------------------------------------------
# The requirement that makes the degradation honest rather than hidden. Three
# conditions, ALL of which must hold for an automatic increase.

def income_evidence_fresh(evidence_tier_code: int, income_staleness_days: int,
                          tables) -> bool:
    """Tier A or B, no older than 45 days."""
    pass


def bureau_view_fresh(bureau_age_days: int,
                      tolerance_days: int = param(35, ge=1, le=120)) -> bool:
    """A bureau view no older than 35 days."""
    return bureau_age_days <= tolerance_days


def expenses_acceptable(expense_basis_code: int, declared_expense_age_months: int,
                        refresh_months: int = param(12, ge=1, le=60)) -> bool:
    """Declared expenses refreshed within 12 months, or replaced by the norm floor."""
    pass


def obligations_stable(obligations_delta_pct: float,
                       growth_ceiling: float = param(0.25, ge=0, le=2.0)) -> bool:
    """Where the bureau shows obligations grown by more than 25% since the last
    assessment, the account goes to the conditional path regardless of income
    tier (s5.6)."""
    return obligations_delta_pct <= growth_ceiling


AutomaticEligible = panel(
    "automatic_eligible",
    members=[module(income_evidence_fresh, name="income_fresh"),
             module(bureau_view_fresh, name="bureau_fresh"),
             module(expenses_acceptable, name="expenses_ok"),
             module(obligations_stable, name="obligations_stable")],
    reduce="all",
    writes={"value": "automatic_path_eligible", "failed": "staleness_codes"},
    evidence=["*"],
)


def affordability_route_code(affordability_verdict_code: int,
                             automatic_path_eligible: bool) -> int:
    """1 automatic, 2 conditional on the client confirming income, 3 fail.

    An account failing affordability on the evidence available is not offered an
    increase at all. An account passing on the evidence available but failing a
    staleness condition is offered one CONDITIONAL on confirmation, and the
    limit may not change until that confirmation is received."""
    pass


Route = module(affordability_route_code, name="affordability_route",
               evidence=["affordability_route_code", "staleness_codes",
                         "affordability_assessment_id"])

Assess = (
    module(notional_instalment_c, name="notional_instalment")
    | ApplyBufferOverlays
    | Affordability
    | AutomaticEligible
    | Route
)
