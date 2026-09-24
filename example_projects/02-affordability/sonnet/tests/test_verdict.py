"""Stage 7 (spec 02 §5.7): the verdict, `indeterminate` distinct from `fail`, and the
three shapes of answer -- standalone, no pipeline (§7.5 style)."""
from decider import Engine, dag

from assessment import verdict


def _pipeline():
    return dag(verdict.evidence_sufficiency_code, verdict.affordability_verdict_code,
               verdict.discretionary_income_after, name="verdict").emit(
        "evidence_sufficiency_code", "affordability_verdict_code", "discretionary_income_after")


def _score(**record):
    base = {
        "income_verification_tier": 2, "minimum_income_tier": 4,
        "applicant_income_evidence_gap": 0, "bureau_is_stale": False, "has_refer_account": False,
        "net_monthly_income": 16000.0, "max_affordable_instalment": 5000.0, "discretionary_income": 8000.0,
    }
    exe = Engine().bind(_pipeline(), mode="interpreted")
    return exe.score({**base, **record})


def test_evidence_sufficiency_is_zero_when_nothing_is_wrong():
    assert _score()["evidence_sufficiency_code"] == verdict.EVIDENCE_OK


def test_applicant_income_gap_is_checked_before_a_stale_bureau_or_a_weak_tier():
    """§5.7.1: named causes, most fundamental first -- an applicant whose income was never
    established at all takes priority over other evidence gaps."""
    out = _score(applicant_income_evidence_gap=1, bureau_is_stale=True, income_verification_tier=6)
    assert out["evidence_sufficiency_code"] == verdict.EVIDENCE_APPLICANT1_INCOME_UNESTABLISHED


def test_stale_bureau_is_indeterminate_not_fail():
    """§4.2: `bureau_is_stale` "produces indeterminate, not fail"."""
    out = _score(bureau_is_stale=True, proposed_instalment=100_000.0)
    assert out["evidence_sufficiency_code"] == verdict.EVIDENCE_BUREAU_STALE
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE


def test_indeterminate_is_never_reachable_through_fail():
    """§5.7.1: "the most consequential error available in this project" -- a huge proposed
    instalment must never turn an evidence gap into `fail`."""
    out = _score(applicant_income_evidence_gap=3, proposed_instalment=1_000_000.0)
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE
    assert out["affordability_verdict_code"] != verdict.FAIL


def test_shape_a_pass_fail_against_a_known_instalment():
    assert _score(proposed_instalment=4000.0)["affordability_verdict_code"] == verdict.PASS
    assert _score(proposed_instalment=5100.0)["affordability_verdict_code"] == verdict.MARGINAL
    assert _score(proposed_instalment=8000.0)["affordability_verdict_code"] == verdict.FAIL


def test_shape_b_capacity_has_no_proposed_instalment_and_only_passes_or_is_indeterminate():
    """§5.7.2(b): "no proposed instalment is supplied and the verdict is pass or
    indeterminate only" -- there is nothing to fail against."""
    ok = _score()
    assert ok["affordability_verdict_code"] == verdict.PASS

    gap = _score(bureau_is_stale=True)
    assert gap["affordability_verdict_code"] == verdict.INDETERMINATE
    # Never FAIL or MARGINAL when there was no instalment to compare against.
    for out in (ok, gap):
        assert out["affordability_verdict_code"] in (verdict.PASS, verdict.INDETERMINATE)


def test_discretionary_income_after_is_none_without_a_proposed_instalment():
    assert _score()["discretionary_income_after"] is None
    assert _score(proposed_instalment=3000.0)["discretionary_income_after"] == 5000.0
