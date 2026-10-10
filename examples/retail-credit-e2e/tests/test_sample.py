"""The shipped sample request, end to end."""
from reference import instalment


def test_the_sample_application_is_approved_for_the_largest_amount_its_instalment_allows(decide, params):
    out = decide()
    assert (out["outcome_code"], out["offer_amount"], out["offer_binding_constraint"]) == (1, 76250.0, "BIND-AFF")
    assert out["instalment"] <= out["instalment_ceiling"] < instalment(params, 76500.0, 60, 7, 5.6, 0.006, 0.006)
    assert out["decline_reason_codes"] == [] and out["primary_reason_code"] is None


def test_the_offer_is_priced_by_hand_to_the_cent(decide, params):
    out = decide()
    assert out["instalment"] == instalment(params, 76250.0, 60, 7, out["credit_life_rate"], 0.006, 0.006)
    assert (out["nominal_annual_rate_unadjusted"], out["nominal_annual_rate"]) == (0.19145, 0.19745)
    assert out["rate_cell_id"] == "rate_card.flex_loan@rc-2026.09#37|39|7"


def test_the_initiation_fee_is_capitalised_so_the_client_finances_more_than_the_advance(decide):
    out = decide()
    assert (out["initiation_fee"], out["initiation_fee_incl_tax"]) == (1480.0, 1702.0)
    assert out["amount_financed"] == out["offer_amount"] + 1702.0


def test_total_cost_is_every_instalment_and_does_not_count_the_capitalised_fee_twice(decide):
    out = decide()
    assert out["total_cost_of_credit"] == round(out["instalment"] * 60, 2)
    assert out["total_interest"] == round(out["instalment_before_fees"] * 60 - out["amount_financed"], 2)
