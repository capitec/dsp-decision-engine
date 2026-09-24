from business_nested import blend, vocab


def test_weights_sum_to_one_and_control_floor_applies():
    r = blend.people_component(
        entity_id=[1, 2], entity_is_owner=[True, False], entity_is_controlling=[True, False],
        entity_relationship_type_code=[vocab.SHAREHOLDER, vocab.DIRECTOR],
        entity_effective_ownership_pct=[70.0, 0.0], entity_criticality_class=[vocab.CRITICAL, vocab.SIGNIFICANT],
        entity_verdict_code=[vocab.CLEAR, vocab.CLEAR], entity_verdict_code_unadjusted=[vocab.CLEAR, vocab.CLEAR],
        entity_pd=[0.02, 0.05], entity_pd_unadjusted=[0.02, 0.05], entity_grade=[3, 5],
    )
    assert abs(sum(r["people_weight_vector"]) - 1.0) < 1e-9
    assert r["people_weight_vector"][0] == 0.7  # >=50% ownership -> weight is its own ownership share


def test_any_disqualifying_entity_declines_the_business_even_if_excluded_from_blend():
    """PP-06/PP-07: exclusion from the average is not exclusion from the rules."""
    r = blend.people_component(
        entity_id=[1, 2], entity_is_owner=[False, True], entity_is_controlling=[False, True],
        entity_relationship_type_code=[vocab.SURETY, vocab.SHAREHOLDER],  # entity 1 is a surety, excluded
        entity_effective_ownership_pct=[0.0, 80.0], entity_criticality_class=[vocab.PERIPHERAL, vocab.CRITICAL],
        entity_verdict_code=[vocab.DISQUALIFYING, vocab.CLEAR],
        entity_verdict_code_unadjusted=[vocab.DISQUALIFYING, vocab.CLEAR],
        entity_pd=[0.9, 0.02], entity_pd_unadjusted=[0.9, 0.02], entity_grade=[12, 2],
    )
    assert r["business_decline_from_entity"] is True
    assert r["business_decline_entity_id"] == 1


def test_insufficient_coverage_below_75_percent():
    """PP-02: scoreable owner ownership / total owner ownership must be >= 75%.
    Entity 2 (60% of the ownership) is unscoreable (grade=None), leaving only 40%
    of the 100% owner ownership scoreable -- well below the 75% requirement."""
    r = blend.people_component(
        entity_id=[1, 2], entity_is_owner=[True, True], entity_is_controlling=[False, False],
        entity_relationship_type_code=[vocab.SHAREHOLDER, vocab.SHAREHOLDER],
        entity_effective_ownership_pct=[40.0, 60.0], entity_criticality_class=[vocab.SIGNIFICANT, vocab.CRITICAL],
        entity_verdict_code=[vocab.CLEAR, vocab.CLEAR], entity_verdict_code_unadjusted=[vocab.CLEAR, vocab.CLEAR],
        entity_pd=[0.02, 0.05], entity_pd_unadjusted=[0.02, 0.05], entity_grade=[3, None],  # entity 2 unscoreable
    )
    assert r["insufficient_people_coverage"] is True
