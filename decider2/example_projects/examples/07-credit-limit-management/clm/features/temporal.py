"""Temporal state: "when did this last change" as an input.

Two things make this harder than it looks.

1. `decision_date` is a SHARED PARAM, not a column (FRAMEWORK-DEMANDS #6). A
   step physically cannot read a per-record date that drifts toward "today",
   which is what makes effective dating unforgeable rather than conventional.

2. Notice periods are in BUSINESS days in the home market and CALENDAR days in
   markets B and C (s6.7). `numpy.busday_offset` is not compilable and carries
   no effective-dated holiday calendar, so the calendar is an artefact.
"""

from decider2 import module, table

CoolingOffWindows = table(
    "clm.cooling_off_windows",
    keys=("change_type_code", "product_code", "path_code"),
    values={"window_months": int},
    dense=True,
    effective_dated=True,
)
# path_code is a KEY, not a branch. s5.11's three carve-outs (X09 at a shorter
# window for client-initiated requests, X11 and X12 not applying at all) are
# rows in this table and in the exclusion panel's `applies_on` column. The two
# paths therefore share one graph and differ only in a declared input --
# which is what makes the s5.12 agreement test a statement about evidence
# rather than about code.

BusinessCalendar = table(
    "clm.business_calendar",
    keys=("jurisdiction_code", "day_index"),
    values={"is_business_day": bool, "business_day_ordinal": int},
    dense=True,
    effective_dated=True,
    description="Holiday calendar per jurisdiction; 20 business days is a lookup.",
)


def months_since_last_limit_change(last_limit_change_day: int, shared) -> int:
    """Whole months between the last applied limit change and decision_date."""
    pass  # (shared.decision_date_day - last_limit_change_day) integer month diff


def months_since_last_offer_response(last_offer_response_day: int, shared) -> int:
    """Whole months since the client last accepted, declined or let an offer lapse."""
    pass


def cooling_off_window_months(
    last_change_type_code: int, product_code: int, path_code: int, tables
) -> int:
    """The window that governs this account's last change, on this path."""
    return tables.cooling_off_windows.window_months[
        tables.cooling_off_windows.cell(last_change_type_code, product_code, path_code)]


def notice_effective_day(
    notice_class_code: int, jurisdiction_code: int, change_type_code: int, shared, tables
) -> int:
    """decision_date plus the prescribed notice period, business or calendar."""
    pass  # calendar lookup for business days; plain addition for calendar days


Temporal = module(
    months_since_last_limit_change,
    months_since_last_offer_response,
    cooling_off_window_months,
    notice_effective_day,
    name="temporal",
    evidence=["cooling_off_window_months", "months_since_last_limit_change"],
)
