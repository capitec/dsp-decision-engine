"""Loop L1 end to end (spec 10 §5.23.1): a client whose actual obligations fail
affordability, whose hypothetical (settle-all-external) obligations pass, gets an offer
priced against the hypothetical -- with the actual/hypothetical distinction preserved
(§5.21.1) and `loop_pass_index`/`loop_termination_code` recorded.
"""
import copy


def _overloaded_record(sample_record: dict) -> dict:
    """A client with heavy *external* obligations (so settling them helps) and a request
    small enough that, once affordable, the solve can find an amount.
    """
    record = copy.deepcopy(sample_record)
    # account_type_code 10 is treated as IMPUTE_PCT_OF_LIMIT (3% of `limit`, core.obligations
    # TREATMENT_MATRIX) and 20 as USE_STATED_INSTALMENT -- both external, so both are settled
    # by this project's simplified P14 (settle every external obligation, 10 §5.15(a-c)).
    record["bureau_accounts"] = [
        {"account_type_code": 10, "balance": 380_000.0, "limit": 400_000.0, "instalment": 4_800.0,
         "months_in_arrears": 0, "opened_date": "2020-01-01", "closed": False, "is_internal": False},
        {"account_type_code": 20, "balance": 60_000.0, "limit": None, "instalment": 5_500.0,
         "months_in_arrears": 0, "opened_date": "2019-05-01", "closed": False, "is_internal": False},
    ]
    record["internal_accounts"] = [
        {"account_type_code": 1, "balance": 5_000.0, "instalment": 300.0,
         "months_in_arrears": 0, "opened_date": "2021-06-15", "closed": False, "is_internal": True},
    ]
    record["requested_amount"] = 30_000.0
    record["term_months"] = 36
    record["is_consolidation_eligible"] = True
    return record


def test_loop_fires_when_actual_affordability_fails_and_scenario_helps(engine, params, sample_record):
    record = _overloaded_record(sample_record)
    out = engine.score(record, params)

    assert out["affordability_verdict_code"] == 3  # FAIL, against actual obligations
    assert out["loop_pass_index"] >= 1
    assert out["existing_obligations_basis_code"] == 2  # ValueBasisCode.HYPOTHETICAL
    # the actual figure is preserved unchanged alongside the hypothetical one (§5.21.1 item 1)
    assert out["existing_obligations"] > out["existing_obligations_hypothetical"]
    assert out["existing_obligations_hypothetical_scenario_ref"] >= 1


def test_loop_does_not_fire_when_affordability_already_passes(engine, params, sample_record):
    out = engine.score(sample_record, params)
    assert out["affordability_verdict_code"] in (1, 2)  # PASS or MARGINAL
    assert out["loop_pass_index"] == 0
    assert out["existing_obligations_basis_code"] == 1  # ValueBasisCode.ACTUAL: never entered the loop


def test_loop_respects_the_four_pass_bound():
    """A scenario where the hypothetical basis still cannot clear the minimum viable
    instalment: the loop must stop at the bound, not run forever."""
    from retail_credit.consolidation import l1_pass
    from retail_credit.vocab import LoopTerminationCode

    state = (0, False, 9_000.0, 0.0, 8_500.0, 8_500.0, 2_000.0, 1_000.0, 0.20, 50_000.0, 60, 12, 60, 40_000.0, 34.0)
    for _ in range(5):
        result = l1_pass(*state)
        loop_pass_index, loop_converged = result[0], result[1]
        state = (loop_pass_index, loop_converged, *state[2:])
        if loop_converged:
            break
    assert loop_pass_index <= 4
    assert loop_converged is True
