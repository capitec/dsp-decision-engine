"""The solve finds the true maximum on the R250 grid, and says which limit stopped it."""
import pytest

from reference import instalment


def reference_best(params, ceiling, term, grade, upper, credit_life_rate=5.6, loading=0.0, addon=0.0):
    """The largest grid amount whose hand-priced instalment fits under `ceiling`, by trying all of them."""
    fits = [a for a in range(2000, int(upper) + 1, 250)
            if instalment(params, float(a), term, grade, credit_life_rate, loading, addon) <= ceiling]
    return float(fits[-1]) if fits else 0.0


def test_the_offer_is_the_largest_amount_that_fits_found_by_trying_every_candidate(decide, params):
    out = decide()
    assert out["offer_amount"] == reference_best(params, out["instalment_cap"], 60, 7, 100_750.0, 5.6, 0.006, 0.006)
    assert out["evaluated"] + out["disqualified"] == 373  # every R250 step from R2 000 to the requested R95 000


@pytest.mark.parametrize("change", [
    {"declared_living_expenses": 9_000.0}, {"requested_amount": 40_000.0, "term_months": 36},
    {"payslip_income": 9_000.0}, {"bureau_score": 800.0, "term_months": 48},
])
def test_other_requests_also_get_the_true_maximum(decide, params, change):
    out = decide(**change)
    loading = 0.0 if out["offer_term_months"] <= 45 else 0.006
    ceiling_amount = min(out["requested_amount"], out["amount_cap"])
    assert out["offer_amount"] == reference_best(
        params, out["instalment_cap"], out["offer_term_months"], out["risk_grade"], ceiling_amount,
        out["credit_life_rate"], loading, 0.006)


def test_a_larger_amount_past_a_band_edge_can_cost_less_per_month(params):
    grade, credit_life_rate = 8, 5.6
    below, edge = (instalment(params, a, 60, grade, credit_life_rate) for a in (59_750.0, 60_000.0))
    assert edge < below  # the card drops about 155 basis points at R60 000


@pytest.mark.parametrize("change, bound", [
    ({"requested_amount": 40_000.0}, "BIND-REQ"),
    ({"requested_amount": 400_000.0, "group_limit": 150_000.0, "group_exposure": 90_000.0, "channel_code": 2}, "BIND-CAP"),
    ({"declared_living_expenses": 9_000.0}, "BIND-AFF"),
])
def test_the_binding_constraint_names_what_stopped_the_amount(decide, change, bound):
    assert decide(**change)["offer_binding_constraint"] == bound


def test_a_request_under_the_product_minimum_is_declined_by_routing_before_any_search(decide):
    out = decide(requested_amount=1_500.0)
    assert out["product_routing_reasons"] == [1201] and out["has_offer"] is False and out["outcome_code"] == 4


def test_a_cap_under_the_product_minimum_leaves_nothing_to_search(decide):
    out = decide(group_limit=148_000.0 + 1_500.0)  # headroom R1 500
    assert (out["solved_amount"], out["solve_binding_constraint"], out["has_offer"]) == (0.0, "BIND-MIN", False)
    assert out["decline_reason_codes"] == [2420] and out["outcome_code"] == 3


def test_the_solve_steps_over_a_band_edge_to_the_amount_that_costs_less_per_month(solve_once, params):
    edge = instalment(params, 60_000.0, 60, 7, 5.6, 0.006)
    assert instalment(params, 59_750.0, 60, 7, 5.6, 0.006) > edge  # the card drops 155 basis points at R60 000
    out = solve_once(instalment_ceiling=edge + 0.01, requested_amount=65_000.0, amount_cap=65_000.0)
    assert out["solved_amount"] == 60_000.0  # R2 000 to R58 250 fit, R58 500 to R59 750 do not, R60 000 does
    assert out["record"]["instalment"] == edge and out["solve_binding_constraint"] == "BIND-AFF"


def test_the_solve_finds_the_true_maximum_for_every_ceiling_on_a_range(solve_once, params):
    for ceiling in range(300, 1_300, 85):
        out = solve_once(instalment_ceiling=float(ceiling), requested_amount=60_000.0, amount_cap=60_000.0)
        assert out["solved_amount"] == reference_best(params, ceiling, 60, 7, 60_000.0, loading=0.006), ceiling


def test_when_nothing_at_the_minimum_fits_under_the_ceiling_there_is_no_winner(solve_once):
    out = solve_once(instalment_ceiling=50.0)
    assert (out["solved_amount"], out["solve_binding_constraint"], out["record"]) == (0.0, "BIND-AFF", None)
