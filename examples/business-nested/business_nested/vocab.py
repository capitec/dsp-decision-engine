"""Codes shared by the modules of this package."""
from __future__ import annotations

# relationship_type_code: how an entity stands to the business.
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

# Most senior first: the role a person keeps when they are reached by more than one path.
ROLE_SENIORITY = (SOLE_PROPRIETOR_PRINCIPAL, MEMBER, PARTNER, TRUSTEE, DIRECTOR, SHAREHOLDER,
                  UBO, SURETY, GUARANTOR, BENEFICIARY, SIGNATORY, GROUP_COMPANY)

OWNING_ROLES = frozenset({MEMBER, SHAREHOLDER, UBO, PARTNER, SOLE_PROPRIETOR_PRINCIPAL})
NON_OWNING_CONTROLLER_ROLES = frozenset({DIRECTOR, TRUSTEE})
# Never part of the people blend: they provide cover or exposure, not people risk.
OUTSIDE_THE_BLEND_ROLES = frozenset({SURETY, GUARANTOR, GROUP_COMPANY, SIGNATORY})

# criticality_class: how much an entity matters to the business.
CRITICAL = 1
SIGNIFICANT = 2
PERIPHERAL = 3

# adverse_verdict_code: what an entity's events add up to.
CLEAR = 1
MINOR = 2
MATERIAL = 3
DISQUALIFYING = 4

# legal_form_code
SOLE_PROPRIETOR = 1


def role_rank(relationship_type_code: int) -> int:
    """Lower is more senior; an unknown role ranks last."""
    return ROLE_SENIORITY.index(relationship_type_code) if relationship_type_code in ROLE_SENIORITY else len(ROLE_SENIORITY)
