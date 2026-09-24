"""Final outcome resolution: combines §5.1 eligibility routing with the search's own
outcome (from `orchestration.py`) into one `outcome_code`/`decline_reason_codes` with a
declared precedence -- the same "every stage runs, the outcome step applies the
correct precedence regardless of what the (in that case meaningless) downstream
stages computed" pattern project 03's own `outcome.py` documents and justifies.
"""
from __future__ import annotations

from decider import step

from consolidation import eligibility, reasons, vocab


def final_outcome_code(
    eligibility_route: str, search_outcome_code: str, search_decline_reason_codes: list[int],
    client_under_administration: bool, income_verified: bool, bureau_unobtainable: bool,
    active_reckless_lending_allegation: bool,
) -> tuple[str, list[int]]:
    """§5.1's routing takes precedence over the search's own outcome: a debt-review
    client is never declined by this flow (CON-ELIG-01's own requirement), and a
    terminal eligibility gate makes the search's result moot regardless of what it
    computed."""
    if eligibility_route == eligibility.ROUTE_DEBT_REVIEW:
        return vocab.OUTCOME_REFER, [reasons.D_DEBT_REVIEW]
    if eligibility_route == eligibility.ROUTE_DECLINE:
        if client_under_administration:
            return vocab.OUTCOME_DECLINE, [reasons.D_ADMINISTRATION]
        if not income_verified:
            return vocab.OUTCOME_DECLINE, [reasons.D_NO_INCOME]
        if bureau_unobtainable:
            return vocab.OUTCOME_DECLINE, [reasons.D_BUREAU_UNOBTAINABLE]
        return vocab.OUTCOME_DECLINE, [reasons.D_NO_INCOME]
    if eligibility_route == eligibility.ROUTE_REFER:
        code = reasons.D_RECKLESS_ALLEGATION if active_reckless_lending_allegation else reasons.D_SERIAL_CONSOLIDATION
        return vocab.OUTCOME_REFER, [code]
    if eligibility_route == eligibility.ROUTE_NOT_APPLICABLE:
        return vocab.OUTCOME_DECLINE, [reasons.D_TOO_FEW_SETTLEABLE]
    return search_outcome_code, search_decline_reason_codes


final_outcome_code_step = step(
    final_outcome_code, outputs=("outcome_code", "decline_reason_codes"),
)
