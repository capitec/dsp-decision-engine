"""P03 hard eligibility — 43 decision points, all evaluated, none short-circuited
before the LAST one. Spec 5.4.

`evaluate_all=True` on the phase envelope (phases/__init__.py) is enforced
here: every gate below runs and records its own verdict, and only the
EXPENSIVE work downstream is short-circuited on a failure. A client who fixes
one problem and re-applies must be told about the second problem the Bank
knew about the first time — so a gate that failed and a gate that was never
evaluated must never look the same in the record.
"""

from __future__ import annotations

from decider2 import module, step

def age_gate(date_of_birth: str, decision_date: str, product_code: int) -> bool:
    """18 minimum; 75-at-maturity unsecured, 70 secured."""
    pass  # compare age at decision_date and at maturity against product bounds

def residency_and_capacity_gate(residency_class: int, capacity_to_contract: bool) -> bool:
    pass  # three residency classes, plus capacity to contract

def product_channel_availability_gate(product_code: int, channel_code: int,
                                      jurisdiction_code: int) -> bool:
    pass  # product available on this channel, in this jurisdiction

def deceased_estate_gate(deceased_flag: bool, estate_flag: bool) -> bool:
    pass  # hard stop on either flag

def debt_review_gate(debt_review_state: int) -> bool:
    """Five states. Degrades separately when the registry is unreachable
    (phases/__init__.py's degradation tuple: cap approvals at R25 000)."""
    pass  # evaluate against the five declared debt-review states

def administration_order_gate(administration_order_flag: bool) -> bool:
    pass  # hard stop

def insolvency_gate(insolvency_state: int) -> bool:
    """Three states."""
    pass  # evaluate against the three declared insolvency states

def sanctions_exclusion_gate(sanctions_hit: bool, internal_exclusion_hit: bool) -> bool:
    """No degraded mode: sanctions list unavailable is a hard stop, all entry points."""
    pass  # evaluate both lists

def staff_restriction_gate(is_staff: bool, product_code: int) -> bool:
    pass  # staff restriction rules by product

def existing_relationship_gate(product_code: int, internal_tenure_months: int) -> bool:
    """Products 11 and 40 only."""
    pass  # require an existing relationship for consolidation and further-advance products

def in_flight_duplicate_gate(client_id: int, decision_date: str) -> bool:
    """Declines a duplicate request within 48 hours. O-24's serialisation point:
    this gate's read must be serialised per client_id, or two simultaneous
    applications from one client can each see the other's absence."""
    pass  # check the in-flight application aggregate, serialised per client (O-24)

# ... 33 further named gates elided here for length; each is one function, one
# verdict, one row in the evaluated set below. None of them short-circuits.

def hard_eligibility_verdict(age_gate: bool, residency_and_capacity_gate: bool,
                             product_channel_availability_gate: bool,
                             deceased_estate_gate: bool, debt_review_gate: bool,
                             administration_order_gate: bool, insolvency_gate: bool,
                             sanctions_exclusion_gate: bool, staff_restriction_gate: bool,
                             existing_relationship_gate: bool,
                             in_flight_duplicate_gate: bool) -> dict:
    """Every gate's verdict, all 43, whether it fired, and the complete reason
    set — never truncated at the first failure (spec 5.4's short-circuit tension)."""
    pass  # assemble the complete reason set; downstream short-circuits, this does not

HardEligibility = module(age_gate, residency_and_capacity_gate,
                         product_channel_availability_gate, deceased_estate_gate,
                         debt_review_gate, administration_order_gate, insolvency_gate,
                         sanctions_exclusion_gate, staff_restriction_gate,
                         existing_relationship_gate, in_flight_duplicate_gate,
                         hard_eligibility_verdict, name="hard_eligibility")
