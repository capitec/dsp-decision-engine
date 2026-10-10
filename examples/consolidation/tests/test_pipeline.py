"""The whole decision on the sample client, pinned, and the params that change it."""
from consolidation.pipeline import build


def test_the_sample_client_is_offered_a_72_month_consolidation_loan_settling_seven_accounts(assess):
    out = assess()
    offer = out["chosen_offer"]
    assert (out["outcome_code"], out["eligibility_route"]) == (2, "proceed")
    assert (offer["scenario_id"], offer["settlement_set"], offer["product_code"], offer["term_months"]) == (
        41, [1, 2, 3, 4, 5, 6, 7], 11, 72)
    assert (offer["offered_amount"], offer["nominal_annual_rate"], offer["instalment"], offer["total_cost_of_credit"]) == (
        211_103.11, 0.1582, 5_117.67, 374_972.24)
    assert (offer["instalment_relief"], offer["total_cost_delta"]) == (1_665.33, 35_692.28)
    assert offer["rate_cell_id"] == "rate_card.product11@rc-p11-2026.09#4|10|6"


def test_the_runners_up_are_materially_different_alternatives(assess):
    out = assess()
    shown = [(out["chosen_offer"]["settlement_set"], out["chosen_offer"]["product_code"]),
             *[(r["settlement_set"], r["product_code"]) for r in out["runners_up"]]]
    assert [r["scenario_id"] for r in out["runners_up"]] == [34, 27] and len(set(map(str, shown))) == 3
    assert all(r["viable"] for r in out["runners_up"])


def test_the_decision_records_the_versions_of_every_table_it_read(assess):
    out = assess()
    assert out["fee_caps_cell_id"] == "fee_caps@fee_caps@2026#caps"
    assert out["statutory_norm_cell_id"] and out["applicant1_tax_table_cell_id"]
    assert out["buffer_grid_cell_id"] and out["residual_floor_cell_id"]
    assert out["reason_registry_version"] == "consolidation-client-2026.09"
    assert out["search_record"]["ordering_rule_set_version"] == "ordering-rules-2026.09"
    assert all(a["settleability_cell_id"] for a in out["classified_accounts"])
    assert {s["rate_card_version"] for s in out["scenarios"] if "rate_card_version" in s} == {
        "rc-p11-2026.09", "rc-p20-promo-2026.09+rc-p20-reversion-2026.09"}


def test_the_statutory_ceiling_and_fee_caps_are_those_in_force_on_the_decision_date(assess):
    assert assess()["pricing_context"] == {
        "statutory_ceiling": 0.28, "initiation_fee_cap": 6_500.0, "monthly_service_fee_cap": 69.0}
    assert assess(decision_date="2025-09-24", bureau_as_of_date="2025-09-20")["pricing_context"]["initiation_fee_cap"] == 6_200.0


def test_the_shipped_params_document_is_the_pipelines_defaults_and_valid(pipeline, documents, params):
    unit = build(*documents)
    defaults = unit.parameters().defaults()
    # the rate cards are documents of their own: the table params only default to their rows
    for key in ("rate_card_product11", "rate_card_product11_version"):
        del defaults["shared"][key]
    for key in ("plain_rate_card", "plain_rate_card_version"):
        del defaults["consolidation"]["plain_granting"][key]
    for key in ("card_promotional_rates", "card_promotional_rates_version", "card_reversion_rates", "card_reversion_rates_version"):
        del defaults["consolidation"]["search"][key]
    assert defaults == params
    unit.parameters().validate(params).check()


def test_a_param_out_of_its_declared_range_is_invalid(documents, params, with_params):
    bad = with_params(params, "flex_loan_policy", anti_harm=0.9)  # the tunable range stops at 60%
    assert build(*documents).parameters().validate(bad).errors


def test_every_params_section_is_named_for_the_team_that_owns_it(params):
    assert {"flex_loan_policy", "everyday_card_policy", "selection", "search", "settlements", "inventory",
            "eligibility_gates", "plain_granting", "outcome"} <= set(params["consolidation"])
    assert set(params["shared"]) == {"overlay_stack_enabled", "credit_life", "credit_life_version"}
    assert params["shared"]["overlay_stack_enabled"] is True


def test_the_sample_request_is_the_documented_shape(sample):
    assert sample["accounts"][0]["account_ref"] == 1 and "application_id" not in sample
