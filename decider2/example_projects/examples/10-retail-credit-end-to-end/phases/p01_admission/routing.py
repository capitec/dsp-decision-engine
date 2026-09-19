"""P01 admission — request validation and routing. Spec 5.2. 41 decision points.

Four steps, none of them credit logic. This is the only phase with no external
dependency (`degradation=None` in phases/__init__.py) and the only place
`decision_date` is ever set. O-04 (ordering.py) makes it immutable from here:
a lint forbids `datetime.now` / `date.today` / `time.time` anywhere under
phases/ and values/, so a phase that wants "today" has already lost.

`Route` is the only place `phase_set_id` is assigned. It is a function of
`entry_point_code`, the named product set, the channel and — for entry point 1
— consolidation eligibility (spec 4.6: `phase_set_id` is NOT derivable from
`entry_point_code` alone). No phase downstream branches on `entry_point_code`
to decide whether it should run; applicability is derived from declared inputs
(entrypoints/manifest.py). This file has no `if entry_point_code == 7`
anywhere, and that absence is the point.
"""

from __future__ import annotations

from decider2 import module, step

# ---------------------------------------------------------------------------
# Normalise.  Eight request shapes -> one internal request.  Zero decision
# points: there is nothing to decide yet, only a shape to agree on.
# ---------------------------------------------------------------------------

def normalised_request(raw_request: dict, entry_point_code: int) -> dict:
    """Collapse one of eight entry-point request shapes into one internal shape."""
    pass  # dispatch on entry_point_code, project into the internal request schema

Normalise = module(normalised_request, name="normalise")

# ---------------------------------------------------------------------------
# Validate.  34 decision points: 14 structural, 11 domain, 9 cross-field.
# Every check runs — a malformed request is rejected wholesale with a
# structured rejection, never partially, and never recorded as a decline.
# ---------------------------------------------------------------------------

def structural_completeness(normalised_request: dict) -> list[str]:
    """14 checks: every field the named product/channel/entry point requires is present."""
    pass  # return missing-field violations, empty if none

def field_domains(normalised_request: dict) -> list[str]:
    """11 checks: term in the product's range, amount above the product minimum,
    decision_date not in the future, channel exists, batch cycle id is open."""
    pass  # return domain violations

def cross_field_consistency(normalised_request: dict) -> list[str]:
    """9 checks: amount/term/product mutually consistent, channel matches the
    presented identity class, campaign id (if any) resolves."""
    pass  # return cross-field violations

def structural_rejection(structural_completeness: list[str], field_domains: list[str],
                         cross_field_consistency: list[str]) -> dict | None:
    """A structured rejection, distinct from a credit decline. Spec 5.2."""
    pass  # non-None iff any of the 34 checks failed; carries which ones

Validate = module(structural_completeness, field_domains, cross_field_consistency,
                  structural_rejection, name="validate")

# ---------------------------------------------------------------------------
# FixDecisionDate.  One decision point, resolved once, for the whole decision.
# O-04.  On a replay it is the RECORDED value, never a fresh clock reading —
# the entire mechanism by which a 2027 decision reproduces in 2034.
# ---------------------------------------------------------------------------

def decision_date(clock_reading: str, replay_decision_date: str | None) -> str:
    """The replayed value if this is a replay, else the request clock, stamped once."""
    pass  # return replay_decision_date if replaying else clock_reading

FixDecisionDate = module(decision_date, name="fix_decision_date")

# ---------------------------------------------------------------------------
# Route.  6 decision points.  Assigns phase_set_id — the one and only place
# it happens.  "Why did phase 14 not run for this client" starts here.
# ---------------------------------------------------------------------------

def candidate_products(normalised_request: dict, entry_point_code: int) -> list[int]:
    """The product set named by the request, or the routed set for batch entry points."""
    pass  # explicit product on 1/2/6/7; a routed candidate set on 3/4/5

def consolidation_eligible(candidate_products: list[int], entry_point_code: int) -> bool:
    """Whether L1 (loops/l1_consolidation.py) is reachable at all this decision.
    A property of the request, not of whether affordability will actually fail."""
    pass  # entry point 1 or 5, existing client, at least one settleable account

def phase_set_id(entry_point_code: int, candidate_products: list[int],
                 channel_code: int, consolidation_eligible: bool) -> int:
    """Assigns phase_set_id from the enumerated table entrypoints.manifest derives
    at build. Archived with the build so `phase_set_id` 6 means the same list in 2034."""
    pass  # look up the enumerated phase set; conditional on consolidation_eligible for EP1

Route = module(candidate_products, consolidation_eligible, phase_set_id, name="route")
