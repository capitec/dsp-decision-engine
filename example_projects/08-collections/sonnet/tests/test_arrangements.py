"""§5.6: arrangements go through project 02's `core.affordability` in ARRANGEMENT mode
(a parameterisation of one capability, never a fork), plus this project's own
sustainability test on top."""
from collections_treatment import arrangements


def test_no_bare_pipeline_module_collision():
    """The whole point of `arrangements._load_affordability_pipeline`: importing this
    module must never populate `sys.modules["pipeline"]`, or `decider`'s own resolution
    of `DECIDER_API__PIPELINE=pipeline:build` risks returning project 02's pipeline
    instead of this project's own. See the module's docstring."""
    import sys
    assert "pipeline" not in sys.modules or sys.modules["pipeline"].__name__ != "_affordability_pipeline_02"
    assert arrangements.ARRANGEMENT == 3


def test_sustainability_passes_when_residual_and_ratio_both_clear_the_floor():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        proposed_arrangement_instalment=500.0, net_monthly_income=10_000.0, discretionary_income=4_000.0,
    )
    assert sustainable is True
    assert residual == 3_500.0
    assert detail == ""


def test_sustainability_fails_below_the_r350_residual_floor():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        proposed_arrangement_instalment=3_900.0, net_monthly_income=10_000.0, discretionary_income=4_000.0,
    )
    assert sustainable is False
    assert "below_minimum_residual" in detail


def test_sustainability_fails_above_85pct_instalment_to_discretionary():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        proposed_arrangement_instalment=3_600.0, net_monthly_income=20_000.0, discretionary_income=4_000.0,
    )
    assert sustainable is False
    assert "instalment_exceeds_discretionary_income_pct" in detail


def test_sustainability_fails_below_5pct_of_net_income_residual():
    sustainable, residual, detail = arrangements.arrangement_sustainability(
        proposed_arrangement_instalment=1_000.0, net_monthly_income=30_000.0, discretionary_income=1_350.0,
    )
    # residual = 350 (clears the R350 floor) but 350/30000 = 1.2% < 5%
    assert sustainable is False
    assert "below_minimum_residual_income_pct" in detail
