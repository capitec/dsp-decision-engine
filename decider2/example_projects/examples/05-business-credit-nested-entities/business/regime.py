"""Stage 5.2 -- regulated or unregulated, and the consequences.

Application grain. Four ordinary steps and one Branch. Nothing here is hard; it
is in the sketch because *what it switches on* reaches into three later stages
and because s5.2's "borderline" flag is the kind of thing that gets dropped.
"""

from decider2 import module, step, param, Branch

REGULATED, UNREGULATED = 1, 2


def is_natural_person_applicant(legal_form_code: int) -> bool:
    """Sole proprietor (1) or partnership of natural persons (2)."""
    pass  # legal_form_code in (1, 2)


def passes_juristic_size_test(
    trailing_turnover: float,
    asset_value: float,
    requested_amount: float,
    size_threshold: float = param(1_000_000.0, ge=0.0,
                                  description="s5.2 turnover AND asset threshold"),
    large_agreement_floor: float = param(250_000.0, ge=0.0,
                                         description="s5.2 large-agreement exemption"),
) -> bool:
    """Both turnover and assets below the threshold, and facility at or below the floor."""
    pass


def regulatory_regime_code(
    is_natural_person_applicant: bool,
    passes_juristic_size_test: bool,
) -> int:
    """s5.2. Natural persons are regulated without a size test."""
    pass


def regime_is_borderline(
    trailing_turnover: float,
    asset_value: float,
    requested_amount: float,
    borderline_band_pct: float = param(10.0, ge=0.0, le=50.0,
                                       description="s5.2: within 10% of a threshold"),
) -> bool:
    """Within 10% of any regime threshold.

    Recorded because a turnover restatement at 5.9 can move an already-priced
    agreement across the line, and the fee cap, the affordability obligation and
    the decline-disclosure obligation all change with it.
    """
    pass


RegulatoryRegime = module(
    is_natural_person_applicant,
    passes_juristic_size_test,
    regulatory_regime_code,
    regime_is_borderline,
    name="regulatory_regime",
    taps=["regulatory_regime_code", "regime_is_borderline"],
    contract="contracts/regulatory_regime.json",
)
