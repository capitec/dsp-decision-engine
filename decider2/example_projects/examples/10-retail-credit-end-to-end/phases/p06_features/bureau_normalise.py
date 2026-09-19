"""P06(a) bureau normalisation — three bureaux, three schemas, one normalised
view. 38 decision points. Spec 5.7(a).

An adverse item that cannot be mapped is a DEFECT, not a silent drop — the
data-quality verdict has four values and an unmappable item forces the worst
of them. This is the file the "no-hit" and "thin-file" segment conditions
(§4.3) are computed from, which is why O-08 (ordering.py) pins it before
segment assignment: segments 1 and 2 are distinguished by thin-file status.
"""

from __future__ import annotations

from decider2 import module, missing_as, param

def header_and_identity(bureau_response: dict) -> dict:
    """May return more than one subject — 0.5% of enquiries. A first-class
    outcome (multi-subject match), not an error."""
    pass  # normalise header block across three bureau schemas

def account_list(bureau_response: dict) -> list[dict]:
    """0..95 entries, median 8, p95 26, p99 47."""
    pass  # normalise the account list, mapping each bureau's own account-type codes

def enquiry_list(bureau_response: dict) -> list[dict]:
    """0..140 entries."""
    pass  # normalise, compute velocity over 30/60/90/365-day windows

def public_record_list(bureau_response: dict) -> list[dict]:
    """0..30 entries."""
    pass  # normalise adverse public records

def data_quality_verdict(account_list: list[dict], enquiry_list: list[dict],
                         public_record_list: list[dict]) -> int:
    """Four values. An adverse item that cannot be mapped to a known type
    forces the WORST value — never a silent drop."""
    pass  # 1 clean .. 4 unmappable-adverse-item-present

def bureau_as_of_date(bureau_response: dict) -> str:
    pass  # the bureau's own as-at date

def bureau_is_stale(bureau_as_of_date: str, decision_date: str,
                    product_code: int, max_age_days: float = param(40.0, ge=0)) -> bool:
    """<= 40 days for 10/11/20/21; <= 25 days for 30/40 (bound narrower by param binding)."""
    pass  # compare (decision_date - bureau_as_of_date) against max_age_days

def no_hit(bureau_response: dict) -> bool:
    """5.8% of entry point 1."""
    pass  # true if the bureau returned no subject

def thin_file(account_list: list[dict]) -> bool:
    """Fewer than three accounts ever, or under 15 months of history. 12.7%.
    Read by segment assignment (P06 §(d)) via O-08."""
    pass  # count accounts and history depth against the thresholds

Normalise = module(header_and_identity, account_list, enquiry_list, public_record_list,
                   data_quality_verdict, bureau_as_of_date, bureau_is_stale,
                   no_hit, thin_file, name="bureau_normalise")
