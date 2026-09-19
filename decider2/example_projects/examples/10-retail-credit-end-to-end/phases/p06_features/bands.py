"""P06(d) derived risk and behavioural features — 31 decision points, ~380
features. Spec 5.7(d). The 36 feature band definitions live here: the codes
the scorecards, the cap register and the campaign trees all key off.

This is the file spec 5.28's worked blast-radius example is about:
`income_band_code` boundary 7 (R28 000 -> R29 500) is edited here and is read
by 9 phases, 31 cap register entries, 4 scorecards and 23 campaign-tree nodes.
navigability/blast_radius.py's static lineage walk starts from a change to
one of these bands.
"""

from __future__ import annotations

from decider2 import module, Table

class FeatureBandTable(Table):
    key: int
    lower_bound: float
    upper_bound: float
    band_code: int

def utilisation_band(revolving_utilisation: float, bands: FeatureBandTable) -> int:
    pass  # bin against the utilisation band table

def arrears_profile(per_account_obligation_annotation: list[dict]) -> dict:
    pass  # arrears counts and recency over 1/3/6/12-month windows

def payment_behaviour(per_account_obligation_annotation: list[dict]) -> dict:
    pass  # on-time / late / missed counts over the same windows

def exposure_ratios(existing_obligations: float, net_monthly_income: float) -> float:
    pass  # existing_obligations / net_monthly_income

def income_band_code(net_monthly_income: float, bands: FeatureBandTable) -> int:
    """One of the 36 band definitions. The spec 5.28 worked example: moving
    boundary 7 from R28 000 to R29 500 reaches 9 phases, 31 cap entries, 4
    scorecards and 23 campaign-tree nodes without anyone enumerating it."""
    pass  # bin net_monthly_income against the income band table

def channel_history(client_id: int) -> dict:
    pass  # channel usage over the client's tenure

def campaign_response_history(client_id: int) -> dict:
    pass  # response history to prior campaign contacts

Band = module(utilisation_band, arrears_profile, payment_behaviour, exposure_ratios,
              income_band_code, channel_history, campaign_response_history, name="bands")
