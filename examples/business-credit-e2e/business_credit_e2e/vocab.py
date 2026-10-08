"""Codes this example declares: the library and the examples it builds on have no vocabulary for a facility's life."""

PRODUCT_TERM = 50
PRODUCT_REVOLVING = 51

# assessment_kind_code
EP1_NEW_TO_BANK = 1
EP3_ANNUAL_REVIEW = 3

# review_basis_code: how much evidence an annual review had. A lower number is better evidence.
REVIEW_COMPLETE = 1
REVIEW_LATE = 2
REVIEW_STALE = 3
REVIEW_TURNOVER_ONLY = 4
REVIEW_NOT_PERFORMED = 5

# comparison_basis_code: whether a decision's grade can be put beside its predecessor's.
COMPARISON_ORIGINATION = 0
COMPARISON_AS_GRADED = 1
COMPARISON_NOT_COMPARABLE = 3

# breach_class_code
BREACH_NONE = 0
BREACH_TECHNICAL = 1
BREACH_MATERIAL = 2
BREACH_SEVERE = 3
COVENANT_NOT_TESTED = -1

# reprice action
REPRICE_NONE = "none"
REPRICE_RESET_ON_NOTICE = "reset_on_notice"
REPRICE_FLAG_RENEGOTIATE = "flag_for_renegotiation"

EXIT_TWO_REVIEWS_AT_GRADE_10 = "grade_10_or_worse_two_consecutive_reviews"
