"""03's own reason code registry (spec 03 §5, §9; 09 §5.15 item 11).

One versioned registry for every reason this project can fire: the 14
eligibility gates, the fraud/consent handling paths, the cap waterfall's
outright declines, affordability, the minimum viable offer rules, and
final validation. Built on `credit_core.reason_codes` (project 00's
mechanism), not re-implemented -- registry, ranking and the primary-reason
pick are exactly what that module already does.
"""
from __future__ import annotations

from credit_core.reason_codes import ReasonCode, ReasonCodeRegistry

# --- Eligibility gates (§5.1) -------------------------------------------------
R_PRODUCT_CHANNEL = 1102
R_MIN_AGE = 1110
R_MAX_AGE_AT_MATURITY = 1111
R_NO_CAPACITY = 1112
R_RESIDENCY = 1120
R_EMPLOYMENT_EXCLUDED = 1130
R_DEBT_REVIEW = 1140
R_ADMIN_ORDER = 1141
R_INSOLVENCY = 1142
R_DECEASED_ESTATE = 1150
R_EXCLUSION_SANCTIONS = 1160
R_EXCLUSION_INTERNAL_FRAUD = 1161
R_EXCLUSION_STAFF = 1162
R_EXCLUSION_LITIGATION = 1163
R_EXCLUSION_OTHER = 1164
R_DUPLICATE_APPLICATION = 1170
R_IN_FLIGHT_APPLICATION = 1171

# --- Consent and fraud handoff (§5.2) -----------------------------------------
R_CONSENT_MISSING = 1201
R_FRAUD_DECLINE = 1210

# --- Cap waterfall outright declines (§5.5) -----------------------------------
R_CHANNEL_CLOSED = 1180
R_EMPLOYMENT_TENURE_TOO_SHORT = 1181
R_ENQUIRY_VELOCITY_EXCESSIVE = 1182
R_GRADE_BEYOND_APPETITE = 1183

# --- Affordability (§5.6) -----------------------------------------------------
R_AFFORDABILITY_FAIL = 1310
R_AFFORDABILITY_INDETERMINATE = 1320

# --- The solve (§5.8) ---------------------------------------------------------
R_EVALUATION_CEILING_REACHED = 1420

# --- Offer-set construction: minimum viable offer rules (§5.9) ---------------
R_BELOW_MIN_AMOUNT = 1601
R_BELOW_MIN_INSTALMENT = 1602
R_TOTAL_COST_RATIO_EXCEEDED = 1603
R_IN_DUPLUM_EXCEEDED = 1604
R_EFFECTIVE_RATE_EXCEEDED = 1605

# --- Final validation (§5.10) -------------------------------------------------
R_FINAL_VALIDATION_FAILED = 1700

# --- Score-driven reasons (§9): project 00's scorecard already owns 4101-4108 --
# reused via `credit_core.scorecard.adverse_action_codes`, not redeclared here.

REGISTRY_VERSION = "loan-granting-reasons-2026.09"

_CODES = [
    (R_FRAUD_DECLINE, 1, "Application-fraud verdict declined", True),
    (R_EXCLUSION_SANCTIONS, 2, "Sanctions list match", True),
    (R_EXCLUSION_INTERNAL_FRAUD, 3, "Internal fraud exclusion list match", True),
    (R_EXCLUSION_STAFF, 4, "Staff restriction list match", True),
    (R_EXCLUSION_LITIGATION, 5, "Litigation exclusion list match", True),
    (R_EXCLUSION_OTHER, 6, "Other exclusion list match", True),
    (R_DECEASED_ESTATE, 7, "Applicant recorded as deceased or under estate administration", True),
    (R_INSOLVENCY, 8, "Sequestrated and not yet rehabilitated", True),
    (R_ADMIN_ORDER, 9, "Active administration order", True),
    (R_DEBT_REVIEW, 10, "Under debt review", True),
    (R_DUPLICATE_APPLICATION, 11, "Duplicate application within 24 hours", True),
    (R_IN_FLIGHT_APPLICATION, 12, "An unconcluded Flex Loan or consolidation application exists", True),
    (R_NO_CAPACITY, 13, "No contractual capacity", True),
    (R_MAX_AGE_AT_MATURITY, 14, "Would exceed the maximum age at maturity", True),
    (R_MIN_AGE, 15, "Below the minimum age", True),
    (R_RESIDENCY, 16, "Residency class not permitted for this product", True),
    (R_EMPLOYMENT_EXCLUDED, 17, "Employment type not permitted for this product", True),
    (R_PRODUCT_CHANNEL, 18, "Product not available on this channel", True),
    (R_GRADE_BEYOND_APPETITE, 20, "Risk grade beyond the Bank's appetite", True),
    (R_CHANNEL_CLOSED, 21, "Channel closed to this product by policy", True),
    (R_EMPLOYMENT_TENURE_TOO_SHORT, 22, "Employment tenure below the minimum policy threshold", True),
    (R_ENQUIRY_VELOCITY_EXCESSIVE, 23, "Excessive recent credit enquiries", True),
    (R_AFFORDABILITY_FAIL, 30, "Proposed instalment exceeds the affordable maximum", False),
    (R_AFFORDABILITY_INDETERMINATE, 31, "Affordability could not be determined", True),
    (R_CONSENT_MISSING, 35, "Bureau-enquiry consent not present or expired", True),
    (R_EVALUATION_CEILING_REACHED, 36, "The solve reached its evaluation ceiling without a proven maximum", True),
    (R_BELOW_MIN_AMOUNT, 40, "Offer amount below the product minimum", False),
    (R_BELOW_MIN_INSTALMENT, 41, "Offer instalment below the minimum viable instalment", False),
    (R_TOTAL_COST_RATIO_EXCEEDED, 42, "Total cost ratio exceeds the policy threshold", False),
    (R_IN_DUPLUM_EXCEEDED, 43, "Scheduled charges would exceed the amount advanced (in duplum)", True),
    (R_EFFECTIVE_RATE_EXCEEDED, 44, "Effective annual rate exceeds the suppression ceiling", False),
    (R_FINAL_VALIDATION_FAILED, 50, "Final validation could not re-derive the offer", True),
]

REGISTRY = ReasonCodeRegistry(REGISTRY_VERSION, [ReasonCode(*row) for row in _CODES])
