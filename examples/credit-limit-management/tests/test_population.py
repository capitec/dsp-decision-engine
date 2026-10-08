import pytest

from limit_mgmt import population


@pytest.mark.parametrize("utilisation, band", [
    (0.0, 1), (0.0001, 2), (0.0999, 2), (0.10, 3), (0.2499, 3), (0.25, 4), (0.40, 5), (0.5499, 5),
    (0.55, 6), (0.70, 7), (0.8999, 7), (0.90, 8), (1.4, 8)])
def test_a_utilisation_on_a_band_edge_falls_in_the_upper_band(utilisation, band):
    assert population.utilisation_band(utilisation) == band


@pytest.mark.parametrize("months, band", [(6, 1), (11, 1), (12, 2), (17, 2), (18, 3), (24, 4), (36, 5), (59, 5), (60, 6), (240, 6)])
def test_a_tenure_on_a_band_edge_falls_in_the_upper_band(months, band):
    assert population.mob_band(months) == band


def test_utilisation_is_the_mean_of_balance_over_limit_across_the_latest_six_cycles():
    balances, limits = [5_000.0, 10_000.0, 0.0, 5_000.0, 5_000.0, 5_000.0, 15_000.0], [10_000.0] * 7
    assert population.revolving_utilisation_6m(balances, limits) == 0.5  # a seventh cycle at 150% would pull it up to 0.64


def test_an_account_with_fewer_than_six_cycles_is_scored_on_what_it_has():
    assert population.revolving_utilisation_6m([2_000.0, 4_000.0], [10_000.0, 10_000.0]) == 0.3
    assert population.cycle_bucket_count([2_000.0, 4_000.0]) == 2


def test_a_cycle_without_a_limit_does_not_count_and_no_history_is_zero_utilisation():
    assert population.revolving_utilisation_6m([5_000.0, 9_000.0], [10_000.0, 0.0]) == 0.5
    assert population.revolving_utilisation_6m([], []) == 0.0


def test_observed_spend_is_the_90th_percentile_of_the_latest_six_months():
    assert population.observed_spend_p90([100.0, 400.0, 200.0, 300.0, 50.0, 250.0, 9_999.0]) == 300.0
    assert population.observed_spend_p90([]) == 0.0
