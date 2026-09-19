"""Project-local vocabulary for the credit limit programme.

Everything the library already names (`decision_date`, `product_code`,
`client_id`, `gross_monthly_income`, `probability_of_default`,
`adjustment_set_id`, `primary_reason_code`, ...) is used unchanged and does not
appear here. This file declares only what 07 owns, and no other project may
redefine these names.

Two conventions are enforced by lint (see FRAMEWORK-DEMANDS #21):

  *_c    an int64 count of cents. Money is never float64 anywhere in this
         project. The artefacts are authored in rand and scaled at load.
  *_code an int8/int16 member of a registered code family below.
"""

from decider2 import Vocabulary, codes, quantised
from decider2.credit import reason_codes

# --- names -----------------------------------------------------------------

vocabulary = Vocabulary(
    owns={
        "account_id": "int64",
        "current_limit_c": "int64",
        "proposed_limit_c": "int64",
        "proposed_limit_unadjusted_c": "int64",
        "applied_limit_c": "int64",
        "additional_limit_c": "int64",
        "months_on_book": "int16",
        "mean_utilisation_6m": "float64",
        "utilisation_band": "int8",
        "mob_band": "int8",
        "observed_spend_p90_c": "int64",
        "behaviour_score": "float64",
        "behaviour_grade": "int8",
        "matrix_cell_id": "int32",
        "matrix_multiplier": "float64",
        "matrix_multiplier_unadjusted": "float64",
        "cycle_dial_id": "int16",
        "binding_cap_code": "int8",
        "evidence_tier_code": "int8",
        "income_staleness_days": "int32",
        "rank_key": "int64",               # quantised ranking value; see ranking.py
        "allocation_rank": "int32",
        "allocation_outcome_code": "int8",
        "change_type_code": "int8",
        "notice_class_code": "int8",
        "offer_id": "int64",
        "consent_record_id": "int64",
        "path_code": "int8",               # 1 programme, 2 client request, 3 simulation
    },
    # The library's `risk_grade` and this project's `behaviour_grade` are
    # different things. Declaring the pair stops a shared module silently
    # binding one to the other (doc 03 s2.1).
    distinct_from={"behaviour_grade": "risk_grade"},
)

# --- code families ---------------------------------------------------------
# Each family is a registered taxonomy: a code, a machine name, a client-facing
# description in three languages, and effective dates. `panel(...)` takes a
# family and maps member module name -> code, so a member cannot exist without
# a code and a code cannot exist without a member.

EXCLUSIONS = codes.family(
    "clm.exclusions", extends=reason_codes.registry,
    members={
        "in_arrears_now": 1, "arrears_within_6m": 2, "debt_review": 3,
        "insolvency": 4, "deceased_or_estate": 5, "fraud_marker": 6,
        "account_dispute": 7, "dormant": 8, "cooling_off": 9,
        "inflight_application": 10, "declined_offer_suppression": 11,
        "no_increase_consent": 12, "staff_account": 13, "too_young": 14,
        "at_product_maximum": 15, "closed_or_arrangement": 16,
    },
)

CAPS = codes.family(
    "clm.caps", extends=reason_codes.registry,
    members={
        "product_maximum": 1, "income_multiple": 2, "total_unsecured": 3,
        "group_exposure": 4, "observed_spend": 5, "matrix_max_increase": 6,
        "affordability": 7,
    },
    # Ties are broken by this order, deterministically (spec s5.5).
    tie_order=["product_maximum", "income_multiple", "total_unsecured",
               "group_exposure", "observed_spend", "matrix_max_increase",
               "affordability"],
)

DECREASE_TRIGGERS = codes.family(
    "clm.decrease_triggers", extends=reason_codes.registry,
    members={f"d{n:02d}": n for n in range(1, 15)},
    attributes={"notice_class": "int8", "severity_rank": "int8"},
)

ALLOCATION_OUTCOMES = codes.family(
    "clm.allocation_outcomes",
    members={
        "funded": 1,
        "below_line": 2,
        "fairness_capped": 3,
        "tail_skipped": 4,
        "overlay_suppressed": 5,      # dropped below the minimum BY an overlay
        "below_minimum": 6,           # dropped below the minimum by the matrix itself
        "not_ranked": 7,
        "conditional_funded": 8,
        "conditional_below_line": 9,
        "client_suppressed": 10,      # a decrease elsewhere on the client (s5.7)
    },
)

ENVELOPES = codes.family(
    "clm.envelopes", members={"limit_budget": 1, "rwa": 2, "expected_loss": 3},
)

# --- exact ordering --------------------------------------------------------
# The ranking value is a float. A float sort key makes the funded set
# irreproducible under re-partitioning, so the sort key is a quantised int64
# and the quantisation is a declared, versioned artefact property, not a
# rounding convention someone remembers.
RANK_KEY = quantised("rank_key", source="ranking_value", scale=1_000_000,
                     mode="half_up", ties=["account_id"])
