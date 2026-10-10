"""The bounded search: a declared budget counted in scenarios, a recorded cause of termination, and the same answer every time."""
import time


def budget(with_params, params, n):
    return with_params(params, "search", max_scenarios=n)


def ids(out):
    return [s["scenario_id"] for s in out["scenarios"]]


def eighteen_accounts(sample):
    """Eighteen settleable accounts: a mix of cards and term loans across providers, all at a high rate."""
    accounts = []
    for i in range(1, 19):
        card, balance, rate, term = i % 3 == 0, 4_000.0 + i * 1_900.0, 0.25 + (i % 5) * 0.012, 48.0 + (i % 3) * 6
        accounts.append({
            "account_ref": i, "account_type_code": 10 if card else 20, "provider_code": (101, 102, 103, 105, 106)[i % 5],
            "is_internal": False, "balance": balance, "nominal_annual_rate": rate, "months_in_arrears": 0,
            "opened_date": "2019-01-01",
            "instalment": round(balance * (rate / 12) / (1 - (1 + rate / 12) ** -term), 2),
            **({"credit_limit": balance * 1.5} if card else {"remaining_term_months": term})})
    return {**sample, "accounts": accounts, "requested_amount": 20_000.0}


def test_the_search_never_exceeds_its_budget_and_says_it_ran_out_of_budget(assess, params, with_params):
    out = assess(params_document=budget(with_params, params, 25))
    record = out["search_record"]
    assert (len(out["scenarios"]), record["budget"], record["consumption"], record["termination_cause"]) == (
        25, 25, 25, "budget_exhausted")
    assert record["candidate_scenarios"] > 25


def test_a_search_that_ran_out_of_candidates_says_so_and_how_many_there_were(assess):
    record = assess()["search_record"]
    assert record["termination_cause"] == "candidates_exhausted"
    assert record["consumption"] == record["candidate_scenarios"] < record["budget"]


def test_a_truncated_search_is_the_first_part_of_the_full_one_and_truncates_identically_on_replay(assess, params, with_params):
    full = assess()
    cut, again = (assess(params_document=budget(with_params, params, 40)) for _ in range(2))
    assert ids(cut) == list(range(1, 41)) == ids(again)
    assert cut["scenarios"] == full["scenarios"][:40] == again["scenarios"]


def test_the_search_does_not_stop_at_the_first_affordable_scenario(assess):
    out = assess()
    viable = [s["scenario_id"] for s in out["scenarios"] if s["viable"]]
    assert len(viable) > 1 and out["chosen_offer"]["scenario_id"] != viable[0]
    assert out["search_record"]["consumption"] > max(viable[:1])


def test_an_eighteen_account_client_is_searched_within_budget_and_the_winner_is_the_same_ten_times(assess, sample, params):
    request = eighteen_accounts(sample)
    start = time.perf_counter()
    first = assess(request)
    elapsed = time.perf_counter() - start
    assert 0 < first["search_record"]["consumption"] <= first["search_record"]["budget"] == 400
    assert first["search_record"]["consumption"] == min(400, first["search_record"]["candidate_scenarios"])
    assert elapsed < 2.5, f"took {elapsed:.2f}s"  # the end-to-end interactive target
    winners = {assess(request)["chosen_offer"].get("scenario_id") for _ in range(9)} | {first["chosen_offer"].get("scenario_id")}
    assert len(winners) == 1 and None not in winners


def test_every_scenario_evaluated_is_recorded_viable_or_rejected_with_a_reason(assess):
    for s in assess()["scenarios"]:
        assert s["viable"] or s["rejection_reason_codes"]


def test_a_different_budget_in_the_params_needs_no_release(assess, params, with_params):
    assert assess(params_document=budget(with_params, params, 7))["search_record"]["consumption"] == 7
