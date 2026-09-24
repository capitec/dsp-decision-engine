"""Shared constants for project 05 (spec 05 §4.2-§4.4, §5.4, §5.6).

Kept separate from `credit_core.vocab` (project 00's own vocabulary module)
because these codes are specific to the nested-entity flow, not shared
library vocabulary -- project 00 does not know entities or events exist.
"""
from __future__ import annotations

# --- relationship_type_code (spec 05 §4.3) -----------------------------------
DIRECTOR = 1
MEMBER = 2
TRUSTEE = 3
BENEFICIARY = 4
SHAREHOLDER = 5
UBO = 6
SURETY = 7
GUARANTOR = 8
GROUP_COMPANY = 9
PARTNER = 10
SOLE_PROPRIETOR_PRINCIPAL = 11
SIGNATORY = 12

# Owning roles (§5.8 PP-02's "owner-entities") -- ownership flows through these.
OWNING_ROLES = frozenset({MEMBER, SHAREHOLDER, UBO, PARTNER, SOLE_PROPRIETOR_PRINCIPAL})
# Non-owning controllers (§5.8 PP-03's "notional weight" roles).
NON_OWNING_CONTROLLER_ROLES = frozenset({DIRECTOR, TRUSTEE})
# Roles PP-01 includes in the blend outright, ownership aside.
BLEND_ROLE_INCLUDE = frozenset({DIRECTOR, MEMBER, TRUSTEE, PARTNER})
# Roles PP-01 excludes from the blend (still subject to disqualification, PP-07).
SURETY_OR_GUARANTOR_ROLES = frozenset({SURETY, GUARANTOR})

# De-duplication seniority order (spec 05 §5.1, most senior first).
ROLE_SENIORITY = (SOLE_PROPRIETOR_PRINCIPAL, MEMBER, PARTNER, TRUSTEE, DIRECTOR, SHAREHOLDER,
                   UBO, SURETY, GUARANTOR, BENEFICIARY, SIGNATORY, GROUP_COMPANY)
_SENIORITY_RANK = {code: i for i, code in enumerate(ROLE_SENIORITY)}


def role_seniority_rank(relationship_type_code: int) -> int:
    """Lower is more senior (spec 05 §5.1's ordering); unknown roles rank last."""
    return _SENIORITY_RANK.get(relationship_type_code, len(ROLE_SENIORITY))


# --- criticality class (spec 05 §5.4) ----------------------------------------
CRITICAL = 1
SIGNIFICANT = 2
PERIPHERAL = 3
CRITICALITY_NAMES = {CRITICAL: "critical", SIGNIFICANT: "significant", PERIPHERAL: "peripheral"}

# --- entity / business adverse verdict (spec 05 §5.6) ------------------------
CLEAR = 1
MINOR = 2
MATERIAL = 3
DISQUALIFYING = 4
VERDICT_NAMES = {CLEAR: "clear", MINOR: "minor", MATERIAL: "material", DISQUALIFYING: "disqualifying"}

# --- outcome (spec 05 §7.1) ---------------------------------------------------
OUTCOME_APPROVE = 1
OUTCOME_APPROVE_WITH_CONDITIONS = 2
OUTCOME_REFER = 3
OUTCOME_DECLINE = 4

# --- segments (reused from credit_core.vocab so calibration keys agree) ------
SEGMENT_SME_PEOPLE = 4     # credit_core.vocab.SEGMENT_SME_PEOPLE
SEGMENT_SME_ENTITY = 5     # credit_core.vocab.SEGMENT_SME_ENTITY

PRODUCT_BUSINESS_TERM_FACILITY = 50
PRODUCT_BUSINESS_REVOLVING_FACILITY = 51

# --- scoring situations (spec 05 §5.7) ---------------------------------------
SCORED = 1
THIN_FILE = 2
NO_HIT = 3
NO_ENQUIRY_POSSIBLE = 4
