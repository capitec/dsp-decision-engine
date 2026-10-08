from limit_mgmt import caps
from limit_mgmt.vocab import Cap

OPEN = dict(cap_product_max=300_000.0, cap_income_multiple=300_000.0, cap_total_exposure=300_000.0,
            cap_observed_spend=300_000.0, matrix_max_increase=300_000.0, matrix_min_increment_unadjusted=500.0)


def propose(**changes):
    return caps.policy_proposed_limit(**{"uncapped_target_limit": 50_000.0, "current_limit": 10_000.0, **OPEN, **changes})


def test_the_limit_is_rounded_down_never_up():
    assert [caps.round_down_500(x) for x in (14_999.0, 15_000.0, 499.0)] == [14_500.0, 15_000.0, 0.0]


def test_the_lowest_cap_wins_and_is_named():
    assert propose(cap_observed_spend=9_600.0 + 10_000.0, cap_income_multiple=22_000.0) == (19_500.0, Cap.C5_OBSERVED_SPEND)


def test_a_tie_between_caps_goes_to_the_lower_code():
    assert propose(cap_product_max=20_000.0, cap_income_multiple=20_000.0) == (20_000.0, Cap.C1_PRODUCT_MAX)


def test_the_matrix_ceiling_is_cap_six():
    assert propose(matrix_max_increase=4_000.0) == (14_000.0, Cap.C6_MATRIX_MAX_INCREASE)


def test_no_cap_binding_when_the_matrix_target_is_the_lowest():
    assert propose(uncapped_target_limit=12_000.0) == (12_000.0, 0)


def test_an_increase_under_the_cells_minimum_increment_is_no_increase():
    assert propose(uncapped_target_limit=11_800.0, matrix_min_increment_unadjusted=2_000.0)[0] == 10_000.0


def test_a_cap_below_the_current_limit_never_lowers_it():
    assert propose(cap_income_multiple=4_000.0)[0] == 10_000.0


def test_observed_spend_holds_a_card_with_tiny_spend_at_its_limit():
    assert caps.cap_observed_spend(current_limit=6_000.0, observed_spend_p90=400.0) == 6_000.0  # 3.5 x 400 = 1 400
    assert caps.cap_observed_spend(current_limit=6_000.0, observed_spend_p90=4_000.0) == 14_000.0


def test_the_exposure_cap_leaves_this_account_only_what_the_clients_other_accounts_do_not_use():
    assert caps.cap_total_exposure(declared_net_income_on_file=17_500.0, total_exposure_other=22_000.0) == 118_000.0
    assert caps.cap_total_exposure(declared_net_income_on_file=100_000.0, total_exposure_other=400_000.0) == 50_000.0
    assert caps.cap_total_exposure(declared_net_income_on_file=10_000.0, total_exposure_other=90_000.0) == 0.0


def test_the_unadjusted_proposal_ignores_the_overlays():
    adjusted = propose(uncapped_target_limit=20_197.5, matrix_max_increase=55_000.0)[0]
    unadjusted = caps.policy_proposed_limit_unadjusted(**{
        "uncapped_target_limit_unadjusted": 21_496.5, "current_limit": 10_000.0, **{k: v for k, v in OPEN.items() if k != "matrix_max_increase"},
        "matrix_max_increase_unadjusted": 55_000.0})
    assert (adjusted, unadjusted) == (20_000.0, 21_000.0)
