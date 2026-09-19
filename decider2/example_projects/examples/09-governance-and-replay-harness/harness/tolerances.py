"""The harness's own tunables - versioned, effective-dated, and governed under
the regime the harness imposes on everything else (spec §9.2).

"A governance harness whose own settings are unversioned is not a governance
harness" (spec §6). So every number in this file is a decider2 param with
declared bounds and an owner class, resolved from an effective-dated document
exactly as a flow's params are, and a change to any of them is a release with
an increment, a declared expected effect and a certification run.

That is not ceremony. A quiet widening of the monetary replay tolerance from
"exact to the cent" to "1e-6 relative" would make a quarter of non-reproducible
decisions reproduce, and the report to the Credit Committee would improve
without anything improving.

OWNERSHIP CLASSES (00 §7.2's three, plus the fourth from 03 §6.3)
  library-global   statutory retention periods         Information Officer
  library-policy   tolerance bands, coverage floors     Decision Governance
  consumer-local   per-flow golden set strata           flow owner
  overlay          - none. The harness has no overlays, deliberately: a
                     governance function that can adjust its own thresholds
                     ad hoc between committee meetings is not one.
"""

from __future__ import annotations

from datetime import date

from decider2 import param
from pydantic import BaseModel, Field


class ReplayTolerances(BaseModel):
    """5 classes x 8 flows. Reviewed annually by Decision Governance."""
    monetary_after_rounding: str = Field("exact_cents", frozen=True)   # not tunable, at all
    rates_decimal_places: int = Field(4, ge=4, le=4)                   # bounds == value, on purpose
    scores_absolute: float = Field(1e-6, ge=1e-9, le=1e-6)
    intermediates_relative: float = Field(1e-12, ge=1e-15, le=1e-12)
    non_reproducible_ceiling: float = Field(0.0005, ge=0.0, le=0.0005)  # 0.05%, Credit Committee


class CoverageFloors(BaseModel):
    rules_exercised: float = Field(0.98, ge=0.95, le=1.0)
    tree_nodes_reached: float = Field(0.95, ge=0.90, le=1.0)
    reason_codes_reachable: float = Field(1.00, ge=1.0, le=1.0)
    dead_rule_days: int = Field(90, ge=30, le=180)
    dominant_rule_share: float = Field(0.40, ge=0.20, le=0.60)


class RetentionPeriods(BaseModel):
    """Enforced BY DELETION, and the deletion is evidenced (spec §5.13.4).
    Keeping evidence longer than the obligation requires is not caution; it is
    a separate compliance problem."""
    credit_decisions_years: int = Field(7, ge=7, le=7)          # statutory floor == ceiling
    marketing_decisions_years: int = Field(5, ge=5, le=5)
    whatif_and_replay_months: int = Field(24, ge=12, le=24)
    monitoring_detail_months: int = Field(13, ge=13, le=13)
    unmasked_access_log_years: int = Field(3, ge=3, le=7)


# Several of these have bounds equal to their value. That is deliberate and it
# is the clearest thing doc 04 §2's validator mechanism does for governance:
# a `Field(7, ge=7, le=7)` is a statutory obligation expressed as a type, and
# the only way to change it is a code change with a code review, which is
# exactly the right friction for a number set by legislation.
#
# The pattern generalises, and the framework docs never mention it:
# VALIDATOR BOUNDS ARE A PERMISSIONS STATEMENT. `ge=6, le=60` on a term cap
# says who may set it and how far. Making the bounds equal says nobody may.
# See FRAMEWORK-DEMANDS D17 - the bounds should carry the OWNER, not only the
# range, because the reviewable artefact needs to print "who may change this"
# beside every value, and today it can only print the range.
