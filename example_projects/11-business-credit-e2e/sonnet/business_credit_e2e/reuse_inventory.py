"""The reuse inventory (spec 11 §5.17) as **real references**, not a table of
claims -- every row below names an actual module/function this project calls
or explicitly declines to call, and every gap names the declared resolution
§5.17.2 requires (extend / compose / parameterise / fork -- never undeclared).

This is SCOPE.md's own words for this slice: "Report the §5.17.7 indicators at
the end. They are this slice's actual deliverable." `report()` computes them
from the data below, not from a narrative -- so a reviewer can check the
arithmetic, not just read the claim.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Component -> disposition (spec 11 §4.9's own table, one row per component
# this project's own README §4.9 names as consumed).
# --------------------------------------------------------------------------

REUSE_TABLE = [
    {"component": "credit_core (project 00), all 22 capabilities", "disposition": "reused_as_is",
     "via": "business_nested.pipeline.build() (05) + covenant.py's direct core.obligations/core.dates calls",
     "note": "Consumed transitively through 05's pipeline for 20 of 22; core.obligations and core.dates "
             "are called directly by this project's own covenant.py."},
    {"component": "project 02 (affordability, 4 modes)", "disposition": "reused_as_is",
     "via": "business_nested.sole_proprietor.py's own importlib-loaded call into assessment/pipeline.py",
     "note": "Not called a second time by this project; the fifth mode §5.17.2 names (periodic "
             "re-assessment of a sole proprietor) is a declared gap -- see GAP_REGISTER."},
    {"component": "project 05 (structure/events/rollup/scoring/blend/financial/grade/pricing/sole-proprietor)",
     "disposition": "reused_as_is", "via": "business_credit_e2e.reuse.load_project05_pipeline().build()",
     "note": "The whole pipeline, one dag call, in both origination.py (EP-1) and review.py (L1). "
             "Zero forks of any of the nine stages."},
    {"component": "project 06 (obligation inventory/settleability, concession catalogue)",
     "disposition": "not_reached",
     "via": "business_credit_e2e.reuse.load_project06_pipeline() exists and loads/binds "
            "(tests/test_reuse.py::test_project06_pipeline_loads), but nothing in this slice calls it",
     "note": "DEPS.md marks 06 hard for project 11, but §4.9 attributes 06's components to L5 only, "
             "and SCOPE.md explicitly skips L4-L6. This slice's own chosen path (EP-1, L1, covenant, "
             "bi-temporal, daily-pass) never reaches a call site for 06. Declared here rather than "
             "forcing a contrived invocation -- see NOTES.md 'Gaps in what I consumed'."},
    {"component": "project 07 (portfolio-budget limit decisioning -- the per-account pipeline)",
     "disposition": "wrapped",
     "via": "review.py::_limit_decision_for_revolving, review.py::_business_facility_as_07_account",
     "note": "07's per-account decider pipeline is called unmodified through Engine().score(); this "
             "project's own field-mapping function (_business_facility_as_07_account) adapts a "
             "business-facility record into 07's retail-shaped input, at working depth -- see "
             "GAP_REGISTER 'limit_mgmt_account_shape'. 07's population-level allocation (§5.8) is not "
             "used at all (a single-facility L1 review is not a portfolio run)."},
    {"component": "project 09 (the 23-item evidence contract)", "disposition": "adopted_as_checklist",
     "via": "NOTES.md's own evidence-contract checklist section",
     "note": "Per SCOPE.md/DEPS.md: 09-C is a requirements document adopted by every project from wave "
             "0, not code. This slice does not implement 09-H (replay/diff/swap-set harness)."},
]

# --------------------------------------------------------------------------
# Declared gaps (spec 11 §5.17.2): component -> what it doesn't produce ->
# the declared resolution (one of extend / compose / parameterise -- never
# fork, per §5.17.2's own four-option table).
# --------------------------------------------------------------------------

GAP_REGISTER = [
    {"id": "sole_proprietor_fifth_mode", "component": "project 02", "gap": "periodic re-assessment mode for a "
     "sole proprietor on year-old evidence (§5.17.2's own named example)", "resolution": "compose",
     "reason": "review.py re-runs 05's existing sole-proprietor call (mode unchanged) on refreshed inputs "
               "rather than building a fifth project-02 mode; the difference between 'origination' and "
               "'periodic re-assessment' evidence rules is not exercised in this slice.",
     "owner": "project 02 (a real fifth mode belongs on its cadence, not forked here)"},
    {"id": "master_scale_version_missing", "component": "project 05", "gap": "05 publishes no "
     "`master_scale_version` field at all, though §5.10.4 requires one on every decision of record",
     "resolution": "compose",
     "reason": "history.master_scale_version_from_result() recovers it from "
               "business_risk_grade_cell_id's own version string (project 00's cell_id convention) "
               "rather than a hand-maintained constant, so it can never drift out of step with the "
               "table that produced the grade -- but it is still this project inferring a field 05 "
               "should be publishing directly.",
     "owner": "project 05 (the natural place to add it, next to risk_grade)"},
    {"id": "product51_pricing_shape", "component": "project 05", "gap": "pricing.py is scoped to a single "
     "product-50 (term loan) lookup; a revolving facility (51) has no instalment in that shape",
     "resolution": "compose",
     "reason": "origination.py's new_facility_instalment() calls project 07's notional_instalment() "
               "(the contractual minimum-payment commitment) for product 51 instead of forking 05's "
               "pricing unit to add a second product shape.",
     "owner": "project 05, if a second facility type becomes common enough to justify a real search"},
    {"id": "limit_mgmt_account_shape", "component": "project 07", "gap": "07's per-account pipeline wants "
     "retail behaviour-score inputs (cycle balances, bureau accounts, utilisation history) this project's "
     "facility record does not carry",
     "resolution": "compose",
     "reason": "_business_facility_as_07_account() maps what a business facility record has (limit, "
               "product code, grade) and leaves the rest at 07's own missing_as() defaults -- a working "
               "approximation, not a business-specific behaviour model.",
     "owner": "project 07 or 11, whichever builds a real business-facility limit-review profile first"},
    {"id": "project06_unreached", "component": "project 06", "gap": "this slice's chosen path never "
     "reaches L5, the only phase §4.9 attributes 06's components to",
     "resolution": "compose",
     "reason": "the loader exists and is tested to load/bind; no call site exists because L4-L6 are "
               "explicitly out of scope (SCOPE.md). Not a code gap so much as a scope gap, recorded "
               "honestly rather than forcing an artificial invocation just to show a non-zero call count.",
     "owner": "whoever builds L5 next"},
]

# --------------------------------------------------------------------------
# Identity-passthrough / relabel count (§5.17.1 item 3): every `.relabel()` or
# pure-passthrough field this project's own code performs on a value it did
# not compute, to carry a published name into a business-flow role. Counted
# by hand against the actual source, not estimated -- each entry names the
# file and line-shape it corresponds to.
# --------------------------------------------------------------------------

PASSTHROUGH_RELABELS = [
    "origination.py: facility_id, knowledge_date, existing_accounts carried through "
    ".emit() untouched (3)",
    "review.py::_business_facility_as_07_account: risk_grade, decision_date relabelled onto "
    "07's per-account field names without renaming their meaning (2)",
    "history.py::new_decision_of_record: facility_id, decision_date, knowledge_date, risk_grade, "
    "probability_of_default carried from the scored record into the decision-of-record unchanged (5)",
]
PASSTHROUGH_RELABEL_COUNT = 10  # 3 + 2 + 5, matched against PASSTHROUGH_RELABELS above


def report() -> dict:
    """The §5.17.7 indicators, computed from this module's own data -- SCOPE.md:
    "Report the §5.17.7 indicators at the end. They are this slice's actual
    deliverable." """
    forks = [r for r in REUSE_TABLE if r["disposition"] == "forked"]
    composed = [g for g in GAP_REGISTER if g["resolution"] == "compose"]
    return {
        "identity_passthrough_relabels": {"value": PASSTHROUGH_RELABEL_COUNT, "threshold": "> 40",
                                           "crossed": PASSTHROUGH_RELABEL_COUNT > 40},
        "gaps_resolved_by_compose": {"value": len(composed), "of_total_gaps": len(GAP_REGISTER),
                                      "threshold": "> half of declared gaps",
                                      "crossed": len(composed) > len(GAP_REGISTER) / 2},
        "forks": {"value": len(forks), "threshold": "> 2", "crossed": len(forks) > 2},
        "releases_blocked_on_cadence": {"value": 0, "threshold": "> 4/year", "crossed": False,
                                         "note": "not measurable from a single build; no release history exists"},
        "locally_retested_consumed_points": {"value": None, "threshold": "> 15% of the 1 280",
                                              "note": "not counted -- this slice does not track the full "
                                                      "1 280-point inventory, only the components it calls"},
        "consumer_side_defects_per_year": {"value": 0, "threshold": "> 6/year",
                                            "note": "no production history exists for this slice"},
    }
