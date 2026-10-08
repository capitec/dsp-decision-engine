"""A search that varies only the debts scores the evidence half once and the capacity half per candidate."""
import pytest

from assessment.pipeline import build, capacity_unit, evidence_unit

CANDIDATES = 400


@pytest.fixture(scope="module")
def halves():
    evidence, capacity = evidence_unit(), capacity_unit()
    return (evidence.compile("interpreted"), evidence.parameters().defaults(),
            capacity.compile("interpreted"), capacity.parameters().defaults())


def with_card_limit(record, limit):
    return {**record, "assessment_mode_code": 4,
            "applicant1_bureau_accounts": [{**record["applicant1_bureau_accounts"][0], "limit": limit}]}


def test_the_evidence_half_does_not_depend_on_the_debts(halves, sample, parse):
    evidence, evidence_params, *_ = halves
    one = evidence.score(parse(sample), evidence_params)
    other = evidence.score(parse(with_card_limit(sample, 90_000.0)), evidence_params)
    assert one["net_monthly_income"] == other["net_monthly_income"] == 16013.5
    assert one["living_expenses"] == other["living_expenses"]
    assert "existing_obligations" not in one


def test_400_candidate_debt_sets_reuse_one_evidence_pass_and_agree_with_the_whole_pipeline(halves, sample, parse, params, pipeline):
    evidence, evidence_params, capacity, capacity_params = halves
    record = parse(sample)
    baseline = {**record, **evidence.score(record, evidence_params)}
    capacities = []
    for i in range(CANDIDATES):
        limit = 1_000.0 + 500.0 * i
        capacities.append(capacity.score(with_card_limit(baseline, limit), capacity_params)["max_affordable_instalment"])
    assert capacities == sorted(capacities, reverse=True)      # more debt, never more capacity
    assert capacities[0] > capacities[-1]
    for i in (0, 137, CANDIDATES - 1):
        whole = pipeline.score(parse(with_card_limit(sample, 1_000.0 + 500.0 * i)), params)
        assert whole["max_affordable_instalment"] == capacities[i]


def test_the_whole_assessment_is_the_two_halves_wired_together(halves, sample, parse, params):
    names = set(build().compile("interpreted").score(parse(sample), params))
    evidence, evidence_params, capacity, capacity_params = halves
    record = parse(sample)
    baseline = {**record, **evidence.score(record, evidence_params)}
    assert names >= set(evidence.score(record, evidence_params)) | set(capacity.score(baseline, capacity_params))
