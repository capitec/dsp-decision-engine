"""Names this project declares locally (spec 11 §4.8) -- the library and projects
00/02/05/06/07 do not publish a business-facility lifecycle vocabulary, so this
project declares its own, once, here. Twenty names in the spec's own count
(§4.8); this file carries the working subset this slice actually uses.
"""
from __future__ import annotations

# --- Products (spec 11 §4.2). 50/51 are this slice's; 52-58 are named but not
# implemented (SCOPE.md: "skip products 52-58"). -------------------------------
PRODUCT_BUSINESS_TERM = 50           # amortising term loan
PRODUCT_BUSINESS_REVOLVING = 51      # revolving, annual review
PRODUCT_CATALOGUE_ONLY = (52, 53, 54, 55, 56, 57, 58)
# Products whose L1 runs a limit decision (§5.4.1 item 3); 54/57 are
# catalogue-only here, so only 51 is ever live in this slice.
REVOLVING_PRODUCTS = frozenset({51, 54, 57})

# --- Entry points (spec 11 §5.1) -- all nine named, only EP-1 and EP-3 built ---
EP1_NEW_TO_BANK = 1
EP2_ADDITIONAL_FACILITY = 2
EP3_ANNUAL_REVIEW = 3
EP4_COVENANT_TEST = 4
EP5_EARLY_WARNING = 5
EP6_AMENDMENT = 6
EP7_RESTRUCTURE_FORBEARANCE = 7
EP8_GROUP_REASSESSMENT = 8
EP9_RM_PREASSESSMENT = 9

ENTRY_POINT_NAMES = {
    EP1_NEW_TO_BANK: "New-to-bank business application",
    EP2_ADDITIONAL_FACILITY: "Additional facility for an existing client",
    EP3_ANNUAL_REVIEW: "Annual review",
    EP4_COVENANT_TEST: "Covenant test",
    EP5_EARLY_WARNING: "Early warning evaluation",
    EP6_AMENDMENT: "Limit or facility amendment",
    EP7_RESTRUCTURE_FORBEARANCE: "Restructure or forbearance",
    EP8_GROUP_REASSESSMENT: "Group exposure re-assessment",
    EP9_RM_PREASSESSMENT: "Relationship-manager pre-assessment",
}

# --- Review basis (spec 11 §5.4.3) -- a mode selector, exactly like project 02's
# four assessment modes (§5.8): one arithmetic path, five declared evidence
# modes. Only 1 (complete) and 2 (late) are exercised by this slice's tests;
# 3-5 are declared (review.py routes to them) but not separately proved. -------
REVIEW_COMPLETE = 1
REVIEW_LATE = 2
REVIEW_STALE = 3
REVIEW_TURNOVER_ONLY = 4
REVIEW_NOT_PERFORMED = 5

# --- Comparability (spec 11 §5.4.2, §5.10.4) -----------------------------------
COMPARISON_AS_GRADED = 1
COMPARISON_RESTATED = 2
COMPARISON_NOT_COMPARABLE = 3
COMPARISON_ORIGINATION = 0  # this project's own extension: no predecessor exists

# --- Grade migration cause decomposition (spec 11 §5.4.1 item 1, six named
# causes; the six must sum to the observed movement, §5.4.1/§5.10.3). ----------
CAUSE_BUSINESS_DATA = 1
CAUSE_ENTITY_STRUCTURE = 2
CAUSE_ENTITY_DATA = 3
CAUSE_MODEL = 4
CAUSE_OVERLAY = 5
CAUSE_SCALE = 6
CAUSE_ORDER = (CAUSE_BUSINESS_DATA, CAUSE_ENTITY_STRUCTURE, CAUSE_ENTITY_DATA,
               CAUSE_MODEL, CAUSE_OVERLAY, CAUSE_SCALE)
CAUSE_NAMES = {
    CAUSE_BUSINESS_DATA: "the business's own data",
    CAUSE_ENTITY_STRUCTURE: "the entity structure",
    CAUSE_ENTITY_DATA: "an entity's own data",
    CAUSE_MODEL: "the model",
    CAUSE_OVERLAY: "an overlay",
    CAUSE_SCALE: "the scheme",
}

# --- Covenant breach classification (spec 11 §5.5.3) ---------------------------
BREACH_NONE = 0
BREACH_TECHNICAL = 1
BREACH_MATERIAL = 2
BREACH_SEVERE = 3
COVENANT_NOT_TESTED = -1  # §5.5.2: "not tested" is a fourth state, neither pass nor breach

# --- Re-pricing action (spec 11 §5.4.1 item 2) ---------------------------------
REPRICE_NONE = "none"
REPRICE_RESET_ON_NOTICE = "reset_on_notice"
REPRICE_AT_NEXT_ROLL = "reprice_at_next_roll"
REPRICE_FLAG_RENEGOTIATE = "flag_for_renegotiation"
REPRICEABLE_ON_NOTICE = frozenset({PRODUCT_BUSINESS_REVOLVING})  # of 50/51, only 51 qualifies

# --- Exit triggers (spec 11 §5.4.1 item 4) -- a working subset, not all 18 -----
EXIT_GRADE_FLOOR = 10  # "grade at or worse than 10 for two consecutive reviews"
EXIT_TRIGGER_GRADE_TWO_REVIEWS = "grade_10_or_worse_two_consecutive_reviews"
EXIT_TRIGGER_SECURITY_COVER = "security_cover_below_0.40_no_remediation"

# --- Bi-temporal structure query (spec 11 §5.13.2) -----------------------------
VIEW_KNOWLEDGE = "knowledge"   # "what did the Bank know when it decided" -- replay
VIEW_EFFECTIVE = "effective"   # "what was actually true then" -- an ownership covenant

# --- Authority (spec 11 §5.14). SCOPE.md: "authority levels beyond two" is out
# of scope -- two of the spec's seven levels are implemented. ------------------
AUTHORITY_L1_CREDIT_ANALYST = 1
AUTHORITY_L2_CREDIT_COMMITTEE = 2
AUTHORITY_AUTOMATED_MANDATE_CEILING = 2_000_000.0  # above this, level 2 is mandatory

# --- Subject kinds (spec 11 §4.1) ----------------------------------------------
SUBJECT_APPLICATION = "application"
SUBJECT_FACILITY = "facility"
SUBJECT_CLIENT = "client"
SUBJECT_GROUP = "group"

# --- Facility status (this project's own -- not in the library) ---------------
FACILITY_STATUS_LIVE = "live"
FACILITY_STATUS_FROZEN = "frozen"
FACILITY_STATUS_CLOSED = "closed"

# --- Master scale version (spec 11 §5.10.4) -- a **declared gap** (§5.17.2):
# project 05 does not publish `master_scale_version` at all (grep confirms no
# such field anywhere in `business_nested/`), even though §5.10.4 requires it
# on "a required output of O9 and a required field of every decision of
# record". Composed around in `history.master_scale_version_from_result()`,
# which recovers it from `business_risk_grade_cell_id`'s own version string
# rather than a hand-maintained constant -- see NOTES.md "Gaps in what I
# consumed".
