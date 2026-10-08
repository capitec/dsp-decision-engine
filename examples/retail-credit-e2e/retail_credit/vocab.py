"""Codes this flow adds to `credit_core.vocab`."""
DEGRADED_NONE = 0
DEGRADED_BUREAU_DOWN = 11  # the one degraded mode this flow models

BIND_AFF = "BIND-AFF"  # the instalment ceiling
BIND_CAP = "BIND-CAP"  # the amount cap
BIND_REQ = "BIND-REQ"  # the requested amount
BIND_MIN = "BIND-MIN"  # nothing at or above the product minimum fits the ceilings
BIND_MAX = "BIND-MAX"  # the product maximum

OUTCOME_REJECTED = 0  # a structural rejection of the request, never a credit decision

BASIS_ACTUAL = 1
BASIS_HYPOTHETICAL = 2  # the client settles their external debts first
