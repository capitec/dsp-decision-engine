"""Project vocabulary. Doc 03 5.2 layer 2, and where it stops being enough.

Layer 2 -- a project-wide name map -- handles the *systematic* differences, and
it does so well. Eight pairs and two prefix families cover every name this
project reads from the library that simply has a different spelling here.

    grep -c '":' vocabulary.py   ->  10

What it CANNOT handle is the case spec 5.17.1 is about: a library name whose
*meaning* depends on the role of the entity it is attached to. `Vocabulary` is a
function from name to name. `applicant_age_years` -> what? It is the director's
age in one place, the surety's age at final instalment date in another, and
undefined for a corporate guarantor. There is no single right-hand side.

A rename cannot express that, and forcing it to try is exactly how doc 01 5.1's
79 identity-passthrough steps happened. So the role-dependent 31 names (spec
5.17.1) are NOT in this file. They are in roles.py, declared once per role
rather than once per name, and the count of them is reported at build.
"""

from decider2 import Vocabulary

CREDIT_CORE = Vocabulary(
    {
        # Systematic spelling differences. Nothing here is role-dependent.
        "client_id": "client_id",
        "total_exposure": "client_exposure_cents",
        "offered_amount": "facility_amount_cents",
        "term_months": "facility_term_months",
        "nominal_annual_rate": "contractual_rate",
        "instalment": "facility_instalment_cents",
        "risk_grade": "business_risk_grade",
        "probability_of_default": "business_pd",
    },
    prefixes={
        "bureau_": "commercial_bureau_",   # the business's own bureau view
        "sector_": "sector_",
    },
    # Money is scaled int64 cents everywhere (doc 03 1.2). The vocabulary
    # enforces the suffix so a `float`-typed money column is a build error and
    # not a reconciliation ticket eighteen months later.
    money_suffix="_cents",
)

# --------------------------------------------------------------------------
# What this project declares locally. Spec 4.8: 20 names, 15 of them about time
# or state over time, and none of the ten earlier specs needed any of them.
#
# The count is the finding, so it is asserted rather than described. Spec 13-Q21
# asks whether lifecycle vocabulary belongs in the library, in a second library,
# or nowhere. This project's answer is in FRAMEWORK-DEMANDS D8: it is a second
# library (`lifecycle-core`), because these names are meaningless to projects
# 01-04 and mandatory for 07, 08 and 11.
# --------------------------------------------------------------------------
LOCAL = Vocabulary.declares(
    "facility_id", "decision_of_record_id", "assessment_kind_code",
    "review_date", "review_basis_code", "covenant_instance_id",
    "covenant_definition_version", "covenant_test_id", "breach_class_code",
    "waiver_id", "watchlist_grade", "signal_set", "cascade_id",
    "allocation_id", "authority_level_code", "master_scale_version",
    "comparison_basis_code", "forbearance_flag", "staging_code",
    "knowledge_date",
    budget=20,     # exceeding this is a build warning naming the new name
)
