"""The daily pass: one entity assessed again must give what a full run gives, on this example's request shape."""
import copy

from business_nested import counterfactual


def test_assessing_only_the_changed_entity_gives_the_full_runs_decision(executable, params, request_for):
    record = request_for()
    original = executable.score(record, params)

    corrected = copy.deepcopy(record)
    corrected["entities"][2]["adverse_events"][0]["is_satisfied"] = True  # Van Wyk's default listing, now paid
    partial = counterfactual.reassess_entity(corrected, original, entity_id=3)
    full = executable.score(corrected, params)

    for name in ("people_pd", "people_grade", "probability_of_default", "risk_grade", "outcome_code", "offered_amount"):
        assert partial[name] == full[name], name
    assert full["people_pd"] < original["people_pd"]  # and the change did matter
