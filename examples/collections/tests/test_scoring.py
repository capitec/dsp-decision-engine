import copy

import pytest
from decikit import ConfigurableStep, DecisionTableConfig, Engine

from collections_treatment import scoring
from collections_treatment.overlays import in_dag
from collections_treatment.pipeline import build

STACKS = ("score_shift", "scaling_change", "odds_multiplier", "band_boundary_shift")


def stacks_off(params, *names):
    off = copy.deepcopy(params)
    for name in names:
        off["collections_treatment"][name]["adjustment_stack_enabled"] = False
    return off


def test_the_sample_account_scores_640_in_band_three(decide):
    out = decide()
    assert (out["collections_score_unadjusted"], out["collections_score"]) == (640.0, 640.0)
    assert out["collections_band_code"] == 3 and out["recovery_curve_cell_id"] == "recovery_curve@recovery-2026.09#3|3"


def test_the_agency_score_shift_lowers_the_score_and_keeps_the_unadjusted_one(decide):
    out = decide(agency_placed_12m=True)
    assert (out["collections_score_unadjusted"], out["collections_score"]) == (640.0, 618.0)
    assert out["score_shift_adjustments_applied"] == ["ADJ-COLL-SCORE-001"]
    assert decide()["score_shift_adjustments_applied"] == []


def test_the_score_shift_lapses_on_its_end_date(decide):
    out = decide(agency_placed_12m=True, decision_date="2027-01-15")
    assert out["collections_score"] == 640.0 and out["score_shift_adjustments_applied"] == []


def test_a_worse_score_is_a_higher_roll_probability_and_can_change_the_band(decide):
    placed, plain = decide(agency_placed_12m=True), decide()
    assert placed["roll_probability_unadjusted"] == plain["roll_probability_unadjusted"]
    assert placed["roll_probability"] > plain["roll_probability"]


def test_the_unadjusted_band_ignores_every_overlay(decide, params):
    off = stacks_off(params, *STACKS)
    for family in (1, 2, 3):
        on = decide(agency_placed_12m=True, product_family_code=family)
        base = decide(params_document=off, agency_placed_12m=True, product_family_code=family)
        assert on["collections_band_code_unadjusted"] == base["collections_band_code"]
        assert on["roll_probability_unadjusted"] == base["roll_probability"]


def test_the_odds_multiplier_applies_to_the_unsecured_book_only(decide):
    unsecured, revolving = decide(product_family_code=1), decide(product_family_code=3)
    assert unsecured["roll_probability"] == pytest.approx(unsecured["roll_probability_before_overlay"] * 1.28)
    assert unsecured["odds_multiplier_adjustments_applied"] == ["ADJ-COLL-SCORE-002"]
    assert revolving["roll_probability"] == revolving["roll_probability_before_overlay"]


def test_the_scaling_change_stretches_the_revolving_scale(decide):
    out = decide(product_family_code=2)
    assert out["adjusted_scale"] == pytest.approx(60.0 * 23 / 20)
    assert out["scaling_change_adjustments_applied"] == ["ADJ-COLL-SCORE-003"]
    assert decide(product_family_code=3)["adjusted_scale"] == 65.0


def test_switching_every_stack_off_reproduces_the_base_run_through_the_same_code(decide, params):
    off = decide(params_document=stacks_off(params, *STACKS), agency_placed_12m=True, product_family_code=1)
    assert off["collections_score"] == off["collections_score_unadjusted"]
    assert off["roll_probability"] == off["roll_probability_unadjusted"]
    assert off["collections_band_code"] == off["collections_band_code_unadjusted"]
    assert [off[f"{name}_adjustments_applied"] for name in STACKS] == [[], [], [], []]


def test_a_stack_adds_its_applied_ids_to_the_response_and_nothing_else(decide):
    out = decide(agency_placed_12m=True)
    assert out["score_shift_adjustments_applied"] == ["ADJ-COLL-SCORE-001"]
    stacks = (*STACKS, "intensity_dial", "treatment_suppression")
    assert not {f"{name}_{part}" for name in stacks for part in ("set_id", "unadjusted")} & set(out)


def test_the_agency_flag_is_read_as_a_flag(pipeline):
    assert next(v.annotation for v in pipeline.plan.inputs if v.name == "agency_placed_12m") == bool | None


def band_of(roll_probability, shift=0.0, unadjusted=None):
    out = Engine().bind(scoring.collections_bands_step).run({
        "roll_probability": roll_probability, "roll_probability_unadjusted": roll_probability if unadjusted is None else unadjusted,
        "band_edge_shift": shift})
    return out["collections_band_code"], out["collections_band_code_unadjusted"]


@pytest.mark.parametrize("probability, bands", [(0.05, 1), (0.10, 2), (0.25, 3), (0.45, 4), (0.65, 5), (0.85, 6), (0.99, 6)])
def test_roll_probability_edges_define_six_bands(probability, bands):
    assert band_of(probability) == (bands, bands)


def test_the_band_boundary_shift_moves_only_the_edge_between_bands_three_and_four():
    assert band_of(0.46, shift=0.03) == (3, 4)  # the 3/4 edge moved up from 0.45 to 0.48
    assert band_of(0.27, shift=0.03) == (3, 3)  # a lower edge did not move with it
    assert band_of(0.66, shift=0.03) == (5, 5)
    assert band_of(0.86, shift=0.03) == (6, 6)


def test_the_band_names_the_cell_it_read(decide):
    assert decide()["collections_band_cell_id"] == "collections_band@collections-band-2026.09#3"


def test_the_band_edges_are_a_shared_param_table(decide, params):
    moved = copy.deepcopy(params)
    moved["shared"]["collections_band_edges"][2]["upper"] = 0.30
    assert decide(params_document=moved)["collections_band_code"] == 4


def test_an_overlay_cannot_name_a_statutory_parameter(stacks):
    document = stacks["score_shift"].model_dump(mode="json")
    document["target"] = document["adjustments"][0]["target"] = "contact_frequency_cap"
    with pytest.raises(ValueError, match="statutory"):
        in_dag(ConfigurableStep.load(document), "applied")
    document["target"] = document["adjustments"][0]["target"] = "collections_score"
    assert len(in_dag(ConfigurableStep.load(document), "applied").adjustments) == 1
    assert in_dag(ConfigurableStep.load(document | {"adjustments": []}), "applied").adjustments == ()


def test_the_pipeline_refuses_a_stack_that_overlays_a_statutory_parameter(stacks, matrix_document):
    document = stacks["score_shift"].model_dump(mode="json")
    document["target"] = document["adjustments"][0]["target"] = "notice_period_days"
    with pytest.raises(ValueError, match="statutory"):
        build(DecisionTableConfig.load(matrix_document), **{**stacks, "score_shift": ConfigurableStep.load(document)})


def test_a_missing_characteristic_scores_its_default_bin_not_zero(decide):
    # 0.4 is the "partial" bin (0 points); no payment at all is -15.
    assert decide(payment_ratio_last=None)["collections_score_unadjusted"] == 625.0
