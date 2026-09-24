"""This project's one declared degraded mode: bureau down (spec 10 §5.25, code 11)."""
import copy


def test_bureau_available_is_not_degraded(engine, params, sample_record):
    out = engine.score(sample_record, params)
    assert out["degraded_mode_code"] == 0
    assert out["source_degradation_codes"] == []


def test_bureau_down_flags_degraded_mode(engine, params, sample_record):
    record = copy.deepcopy(sample_record)
    record["bureau_available"] = False
    out = engine.score(record, params)
    assert out["degraded_mode_code"] == 11
    assert 11 in out["source_degradation_codes"]


def test_bureau_down_within_the_reduced_envelope_still_approves(engine, params, sample_record):
    record = copy.deepcopy(sample_record)
    record["bureau_available"] = False
    record["requested_amount"] = 20_000.0
    record["internal_tenure_months"] = 30.0
    # `worst_arrears_months` is P06-computed (from `bureau_accounts`/`internal_accounts`),
    # never a request field -- the sample record's accounts already carry none in arrears.
    # segment must land in 3-5 for the envelope: existing, credit-holding, clean.
    out = engine.score(record, params)
    assert out["degraded_mode_code"] == 11
    if out["bureau_down_envelope_ok"]:
        assert out["outcome_code"] != 4  # not a decline, at least


def test_bureau_down_outside_the_reduced_envelope_refers(engine, params, sample_record):
    record = copy.deepcopy(sample_record)
    record["bureau_available"] = False
    record["requested_amount"] = 95_000.0  # above the R25 000 reduced-envelope ceiling
    out = engine.score(record, params)
    assert out["degraded_mode_code"] == 11
    assert out["bureau_down_envelope_ok"] is False
    assert out["outcome_code"] == 3  # refer, never a silent approval, never a decline
